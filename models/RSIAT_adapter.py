import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
import math
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None

        # SD-RSIAT (flag OFF by default -> original RSIAT path, nothing below is used)
        self._sdrsiat = bool(args.get("sdrsiat", False))
        self.sd = None
        if self._sdrsiat:
            from models.sdrsiat import SDRSIATController
            self.sd = SDRSIATController(args)

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)
            
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

      
        if self._cur_task >0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)

        if self._sdrsiat:
            self.sd.begin_task(self._sdrsiat_adapters())
        self._train(self.train_loader, self.test_loader)
        if self._sdrsiat:
            self.sd.end_task(self._sdrsiat_adapters(), self._cur_task)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        task_size = data_manager.get_task_size(self._cur_task)

        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

        if self._sdrsiat:
            # after EVERY task (incl. the first): second-moment stats of current-task data
            self._sdrsiat_collect_statistics(data_manager)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            if self._sdrsiat:
                self._sdrsiat_inc_train(train_loader, test_loader)
                return
            param_groups = []
            param_groups.append(
                {'params': self._network.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'], 'weight_decay': self.args['ae_weight_decay']})
            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _inc_loss(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):     
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            loss_base = lambda_rs * self.rs_loss_func(features, targets)
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_inc = self._inc_loss(features, features_old)
        return logits, loss_c, loss_inc

    # ------------------------------------------------------------------
    # SD-RSIAT (only reachable when args['sdrsiat'] is True)
    # ------------------------------------------------------------------
    def _sdrsiat_adapters(self):
        from models.sdrsiat import get_adapters
        net = self._network.module if hasattr(self._network, "module") else self._network
        return get_adapters(net.convnet)

    def _sdrsiat_collect_statistics(self, data_manager):
        """Eval-mode, no_grad pass over the CURRENT task's train split (test-time
        transforms, no augmentation). Hooks read the inputs of down_proj (x) and
        up_proj (h = ReLU(.), dropout inactive in eval); token set = stats_token_mode."""
        from models.sdrsiat import collect_adapter_statistics
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.to(self._device)
        adapters = self._sdrsiat_adapters()
        stats_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="test")
        stats_loader = DataLoader(stats_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        C_in, C_hid, info = collect_adapter_statistics(
            net, stats_loader, adapters, self._device, token_mode=self.sd.stats_token_mode,
            max_batches=self.sd.stats_max_batches, forward_fn=net.extract_vector)
        logging.info("[SD-RSIAT] task {} stats tokens (layer0): in={} hid={}".format(
            self._cur_task, info["tokens_in"][0] if info["tokens_in"] else 0,
            info["tokens_hid"][0] if info["tokens_hid"] else 0))
        self.sd.update_statistics(C_in, C_hid, self._cur_task)

    def _inc_loss_parts(self, features, features_old):
        """Same math as _inc_loss, but returns (L_align, L_orth) separately (unweighted)."""
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device, non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])
        return loss_align, loss_orth

    def _sdrsiat_compute_losses(self, inputs, targets):
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
        with torch.no_grad():
            # old network is frozen (requires_grad=False, eval) -> identical values to the original path
            features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_align, loss_orth = self._inc_loss_parts(features, features_old)
        return logits, loss_c, loss_align, loss_orth

    def _sdrsiat_inc_train(self, train_loader, test_loader):
        from models.sdrsiat import get_adapters, route_gradients
        net = self._network_module_ptr
        adapters = get_adapters(net.convnet)
        roles = self.sd.split_roles(adapters)
        adapter_ids = {id(p) for p in roles["all_adapter"]}
        other = [p for p in net.convnet.parameters() if p.requires_grad and id(p) not in adapter_ids]
        if other:
            logging.warning("[SD-RSIAT] {} non-adapter trainable convnet params: routed like original "
                            "(L_cos + beta*L_align), not projected".format(len(other)))
        fc_params = [p for p in net.fc.parameters() if p.requires_grad]
        proj_params = [p for p in self.old_ae.parameters() if p.requires_grad]

        param_groups = self.sd.build_param_groups(
            roles, base_lr=self.init_lr, base_wd=self.weight_decay,
            fc_params=fc_params, fc_lr=self.init_lr, fc_wd=self.weight_decay,
            proj_params=proj_params, proj_lr=self.args['ae_init_lr'], proj_wd=self.args['ae_weight_decay'],
            other_params=other)
        required = [p for p in roles["all_adapter"] if p.requires_grad] + other + fc_params + proj_params
        self.sd.check_param_groups(param_groups, required=required, excluded=roles["frozen"])
        for g in param_groups:
            logging.info("[SD-RSIAT] group {}: {} tensors, lr={:.3g}, wd={:.3g}".format(
                g["name"], len(g["params"]), g["lr"], g["weight_decay"]))

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer {}".format(self.args['optimizer']))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
        log_count_parameter(param_groups)
        self.sd.build_projectors(self._device)

        beta, gamma = self.args["beta"], self.args["gamma"]
        prog_bar = tqdm(range(self.tuned_epochs))
        info = ""
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_c, losses_rt = 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_align, loss_orth = self._sdrsiat_compute_losses(inputs, targets)
                optimizer.zero_grad(set_to_none=True)
                route_gradients(loss_c, loss_align, loss_orth, beta, gamma,
                                stable_params=roles["stable"], plastic_params=roles["plastic"],
                                projector_params=proj_params, classifier_params=fc_params,
                                other_params=other,
                                route_align_to_plastic=self.sd.route_align_to_plastic)
                snaps = self.sd.snapshot(adapters)
                optimizer.step()
                self.sd.project_updates(adapters, snaps)

                loss_rt = beta * loss_align.detach() + gamma * loss_orth.detach()
                losses += (loss_c.detach() + loss_rt).item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "[SD-RSIAT] Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.tuned_epochs, losses / len(train_loader),
                losses_c / len(train_loader), losses_rt / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = torch.matmul(features, features.t())

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss