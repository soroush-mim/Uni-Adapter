from this import d
import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
import os
import clip
import open_clip

# Fix imports based on your structure
from utils.utils import scaled_all_reduce, AverageMeter, ProgressMeter, accuracy
import utils.math_utils as math_utils
from visualize.visualization import visualize_pointclouds_plotly

from dota import DOTA
from dota_mixture import DOTA_mix


# Use math logic directly
def softmax_entropy(x, enable_softmax=True, temperature=1.0):
    if enable_softmax:
        probs = torch.softmax(x / temperature, dim=1) 
    else:
        probs = x
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1)

def get_entropy(loss, clip_weights):
    max_entropy = np.log2(clip_weights.size(1))
    return (loss / max_entropy).type(torch.float32)

@torch.no_grad()
def clip_classifier(args, classnames, template, clip_model):
    clip_weights = []
    for classname in classnames:
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        
        if args.vlm3d == 'uni3d' or args.vlm3d == 'ulip':
            texts = clip.tokenize(texts).cuda()
        elif args.vlm3d == 'openshape':
            texts = open_clip.tokenizer.tokenize(texts).cuda()
            
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        clip_weights.append(class_embedding)

    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def get_logits_wrapper(args, model, feature, clip_weights):
    if args.vlm3d == 'uni3d':
        pc_features = model.encode_pc(feature)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        logits = 40. * pc_features @ clip_weights ##### should 40 multiplier be applied to other models logits?
    elif args.vlm3d == 'ulip':
        xyz = feature[:, :, :3]
        pc_features = model(xyz)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        logits = 100. * pc_features @ clip_weights
    elif args.vlm3d == 'openshape':
        xyz = feature[:, :, :3]
        pc_features = model(xyz, feature)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        logits = 100. * pc_features @ clip_weights
    
    loss = softmax_entropy(logits)
    prob_map = logits.softmax(1)
    #pred = logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0].type(torch.int32)
    pred = int(logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0].type(torch.int32))

    
    return pc_features, logits, loss, prob_map, pred

# --- UPDATED CACHE FUNCTION (Single Cache) ---
def update_cache(cache, pred, features_loss, shot_capacity, clip_weights, beta=150):
    pc_features, entropy, prop_entropy, prob_map = features_loss
    confidence = torch.exp(-beta * prop_entropy)
    item = [pc_features, confidence, prob_map]
    
    add_new_center = False
    
    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item + [1]) # [feat, conf, prob, count]
            add_new_center = True
        else:
            add_new_center = False
            # Cluster similarity merge logic (Prototype Reassignment)
            cluster_similarities = []
            for cluster_item in cache[pred]:
                feat_cluster, _, _, _ = cluster_item
                sim = pc_features @ feat_cluster.T
                cluster_similarities.append(sim)
            
            cluster_similarities = torch.stack(cluster_similarities)
            max_idx = torch.argmax(cluster_similarities)
            
            feat_c, conf_c, prob_c, count_c = cache[pred][max_idx]
            
            # Weighted update
            new_feat = (conf_c * count_c * feat_c + confidence * pc_features) / (count_c * conf_c + confidence)
            new_feat = new_feat / new_feat.norm(dim=-1, keepdim=True)
            new_count = count_c + 1
            
            new_prob = torch.softmax(100 * new_feat @ clip_weights, dim=1)
            new_loss = softmax_entropy(100 * new_feat @ clip_weights)
            new_prop_ent = torch.tensor(get_entropy(new_loss, clip_weights), device=pc_features.device)
            new_conf = torch.exp(-beta * new_prop_ent)
            
            cache[pred][max_idx] = [new_feat, new_conf, new_prob, new_count]
    else:
        add_new_center = True
        cache[pred] = [[pc_features, confidence, prob_map, 1]]

    return add_new_center

def compute_cache_logits(pc_features, cache, clip_weights, prev_info, iteration, hp):
    cache_keys = []
    all_probs = []
    
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            feat, _, prob, _ = item
            cache_keys.append(feat)
            all_probs.append(prob)

    if not cache_keys:
        return torch.zeros((pc_features.size(0), clip_weights.size(1)), device=pc_features.device), prev_info

    all_probs = torch.cat(all_probs, dim=0) 
    cache_keys = torch.cat(cache_keys, dim=0)

    # Graph-based Label Smoothing
    add_new_center, L_reg_old, L_reg_old_inv = prev_info
    
    all_probs_new, new_info = math_utils.online_value_refinement_new(
        cache_keys, all_probs, 
        add_new_center, L_reg_old, L_reg_old_inv, 
        iteration, 
        threshold=hp['threshold'], 
        lambda_reg=hp['lambda_reg'],
        k=1
    )

    _, new_classes = torch.max(all_probs_new, dim=1)
    cache_values = F.one_hot(new_classes, num_classes=clip_weights.size(1)).float()
    cache_counts = cache_values.sum(dim=0) + 1e-6
    cache_values = cache_values / cache_counts
    
    pc_features = F.normalize(pc_features, dim=-1)
    affinity = (pc_features @ cache_keys.permute(1, 0))
    cache_logits = affinity.to(dtype=cache_values.dtype) @ cache_values 
    
    return cache_logits, new_info

def compute_cache_logits_old(pc_features, cache, clip_weights, hp):
    cache_keys = []
    all_probs = []
    
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            feat, _, prob, _ = item
            cache_keys.append(feat)
            all_probs.append(prob)

    if not cache_keys:
        return torch.zeros((pc_features.size(0), clip_weights.size(1)), device=pc_features.device)

    all_probs = torch.cat(all_probs, dim=0) 
    cache_keys = torch.cat(cache_keys, dim=0)

    all_probs_new = math_utils.online_value_refinement_old(
        cache_keys, all_probs, 
        threshold=hp['threshold'], 
        lambda_reg=hp['lambda_reg']
    )

    _, new_classes = torch.max(all_probs_new, dim=1)
    cache_values = F.one_hot(new_classes, num_classes=clip_weights.size(1)).float()
    cache_counts = cache_values.sum(dim=0) + 1e-6
    cache_values = cache_values / cache_counts
    
    pc_features = F.normalize(pc_features, dim=-1)
    affinity = (pc_features @ cache_keys.permute(1, 0))
    cache_logits = affinity.to(dtype=cache_values.dtype) @ cache_values 
    return cache_logits

def test_zeroshot_3d_core(test_loader, validate_dataset_name, model, clip_model, tokenizer, args, hp):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, top3, top5], prefix='Test: ')

    model.eval()
    
    # DOTA/MODE-DOTA Configuration
    if args.use_dota or args.use_mode_dota:
        dota_cfg = {
            'epsilon': args.dota_epsilon,
            'sigma': args.dota_sigma,
            'eta': args.dota_eta,
            'rho': args.dota_rho
        }
    if args.use_dota:
        logging.info(f"DOTA Hyperparameters: {dota_cfg}")
    if args.use_mode_dota:
        logging.info(f"MODE DOTA Hyperparameters: {dota_cfg}")
        logging.info(f"num MODES for MODE DOTA: {args.mode_M}")

    with torch.no_grad():
        logging.info('=> Encoding text anchors')
        
        if args.precomputed_text_features and os.path.exists(args.precomputed_text_features):
            logging.info(f"Loading precomputed text features from {args.precomputed_text_features}")
            text_features = torch.load(args.precomputed_text_features, map_location=args.device, weights_only=True)
        else:
            if not os.path.exists(args.templates_path):
                raise FileNotFoundError(f"Templates not found at {args.templates_path}")
            if not os.path.exists(args.labels_path):
                raise FileNotFoundError(f"Labels not found at {args.labels_path}")
                
            with open(args.templates_path) as f:
                templates = json.load(f)[args.template_key] 
            with open(args.labels_path) as f:
                labels = json.load(f)[validate_dataset_name]
                
            logging.info("Computing text features on the fly...")
            text_features = clip_classifier(args, labels, templates, clip_model)
            # torch.save(text_features.cpu(), args.precomputed_text_features)

        text_features = text_features.to(args.device)
        
        
        input_shape_for_adaptation_models = text_features.shape[1] if args.vlm3d == 'uni3d' else text_features.shape[0]
        # num_classes_for_adaptation_models is derived from text_features, which is reliable.
        num_classes_for_adaptation_models = text_features.shape[0] if args.vlm3d == 'uni3d' else text_features.shape[1] # clip_weights.shape[1]

        # Initialize DOTA model if enabled
        dota_model = None
        # Initialize GMM-DOTA model if enabled
        mode_dota_model = None

        if args.use_dota and not args.use_mode_dota: # Only DOTA is enabled
            tensor_matrix = torch.full((input_shape_for_adaptation_models, num_classes_for_adaptation_models), 0.001)
            dota_model = DOTA(dota_cfg, input_shape_for_adaptation_models, num_classes_for_adaptation_models, tensor_matrix)
            dota_model.eval() # DOTA is used during inference/TTA, not trained in a traditional sense.
            logging.info("Initialized DOTA model.")

        elif args.use_mode_dota:
            mode_dota_model = DOTA_mix(dota_cfg, input_shape_for_adaptation_models, num_classes_for_adaptation_models, text_features.T, num_modes=args.mode_M)
            mode_dota_model.eval()
            logging.info(f"Initialized MODE-DOTA model with M={args.mode_M}.")


        # --- Uni-Adapter Cache Initialization (Only if DOTA/GMM-DOTA are not used) ---
        if not args.use_dota and not args.use_mode_dota:
            cache = {} 
            L_reg_old = 0 
            L_reg_old_inv = 0

        stored_times = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            
            # --- Visualization ---
            if i == 0 and args.output_dir:
                try:
                    vis_path = os.path.join(args.output_dir, f'vis_batch_{i}')
                    viz_dict = {f"Sample_{j}_{target_name[j]}": pc[j].cpu().numpy() for j in range(min(2, len(pc)))}
                    visualize_pointclouds_plotly(viz_dict, save_path=vis_path, title=f"Test Batch {i} Input")
                except Exception as e:
                    logging.warning(f"Visualization failed: {e}")

            torch.cuda.synchronize()
            start_event.record()

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            target = target.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb), dim=-1)

            if args.vlm3d == 'uni3d':
                clip_weights = text_features.float().t()
            else:
                clip_weights = text_features.float()
                model = model.float()

            # A. Get Base Logits
            pc_features, clip_logits, loss, prob_map, pred = get_logits_wrapper(args, model, feature, clip_weights)
            
            # --- DOTA Logic ---
            if args.use_dota and not args.use_mode_dota and dota_model is not None:
                dota_logits = dota_model.predict(pc_features.mean(0).unsqueeze(0).half())
                dota_model.fit(pc_features, prob_map)
                dota_model.update()

            elif args.use_mode_dota and mode_dota_model is not None:

                dota_logits = mode_dota_model.predict(pc_features.mean(0).unsqueeze(0).half())
                mode_dota_model.fit(pc_features, prob_map)

                # Augment point cloud by adding Gaussian noise, then get its feature
                noise_std = 0.05  # Standard deviation for augmentation, can adjust as needed
                pc_aug = pc + noise_std * torch.randn_like(pc)
                feature_aug = torch.cat((pc_aug, rgb), dim=-1)
                
                pc_features_aug, _, _, prob_map_aug, _ = get_logits_wrapper(args, model, feature_aug, clip_weights)

                # pc_features_interpolated = (0.5*pc_features + 0.5*pc_features_aug) 
                # #normalize the pc_features_interpolated
                # pc_features_interpolated = pc_features_interpolated / pc_features_interpolated.norm(dim=-1, keepdim=True)
                pc_features_aug = pc_features_aug / pc_features_aug.norm(dim=-1, keepdim=True)
                mode_dota_model.fit(pc_features_aug, prob_map)
                # mode_dota_model.fit(pc_features_interpolated, prob_map)


                # mode_dota_model.fit(pc_features_aug2, prob_map)
                # prop_entropy = torch.tensor(get_entropy(loss, clip_weights), device=args.device)
                # dota_beta = 20.0 # args.dota_beta
                # confidence_weight = torch.exp(-dota_beta * prop_entropy).unsqueeze(1)
                # weighted_prob_map = prob_map * confidence_weight
                # mode_dota_model.fit(pc_features, weighted_prob_map)

                mode_dota_model.update()
                

                # Combine clip_logits and dota_logits
                # DOTA's original logic for dota_weights
                dota_weights_val = torch.clamp(dota_cfg['rho'] * mode_dota_model.c.mean() / pc_features.size(0), max=dota_cfg['eta'])
                # dota_weights_val = torch.clamp(dota_cfg['rho'] * mode_dota_model.c.mean() / pc_features.size(0), max=1e10)
                # dota_weights_val = dota_cfg['rho']
                # prior = torch.log(((mode_dota_model.c.sum(dim=1))/mode_dota_model.c.sum()))
                # final_logits = clip_logits + dota_weights_val * dota_logits
                dota_logits = dota_weights_val * (dota_logits) 
                # dota_logits = dota_logits-dota_logits.min()
                # rescale the dota_logits to have the same min and max as clip_logits
                # dota_logits_min = dota_logits.min()
                # dota_logits_max = dota_logits.max()
                # clip_logits_min = clip_logits.min()
                # clip_logits_max = clip_logits.max()
                # dota_logits = (dota_logits - dota_logits_min) / (dota_logits_max - dota_logits_min) * (clip_logits_max - clip_logits_min) + clip_logits_min
               # combine weighted by inverse of entropy
                entropy_clip = softmax_entropy(clip_logits)
                entropy_dota = softmax_entropy(dota_logits)
                weight_clip = 1/(entropy_clip+1e-3)
                weight_dota = 1/(entropy_dota+1e-3)
                weight_clip = weight_clip/(weight_clip+weight_dota)
                weight_dota = weight_dota/(weight_clip+weight_dota)
                # print("entropy_clip", entropy_clip)
                # print("entropy_dota", entropy_dota)
                # print("....................................")
                # final_logits = (1/(entropy_clip+1e-3)) * clip_logits + (1/(entropy_dota+1e-3)) * dota_logits
                final_logits = weight_clip * clip_logits + weight_dota * dota_logits
                # final_logits = (1-(0.1*dota_weights_val/0.1))*clip_logits + (0.9*dota_weights_val/0.1)*dota_logits
                # import matplotlib.pyplot as plt
                # plt.close()
                # plt.plot(clip_logits.cpu().numpy()[0], label='clip_logits')
                # plt.plot(dota_logits.cpu().numpy()[0], label='dota_logits')
                # plt.legend()
                # plt.savefig('dota_logits.png')
                # plt.close()
                # import time
                # time.sleep(1)
                # clip_prob = clip_logits.softmax(1).cpu().numpy()
                # dota_prob = (dota_weights_val *dota_logits).softmax(1).cpu().numpy()
                # clip_logits_norm = (clip_logits - clip_logits.min()) / (clip_logits.max() - clip_logits.min())
                # dota_logits_norm = (dota_logits - dota_logits.min()) / (dota_logits.max() - dota_logits.min())
                # final_logits = clip_logits_norm + dota_logits_norm
                # print(prob_map.max())
               

            # --- Original Uni-Adapter Cache Logic ---
            elif not args.use_dota and not args.use_gmm_dota: # if neither DOTA nor GMM-DOTA are used
                # Normalize entropy to [0, 1]
                prop_entropy = torch.tensor(get_entropy(loss, clip_weights), device=args.device)

                # B. Update Cache (Using Single Cache)
                add_new_center = update_cache(
                    cache, pred, 
                    [pc_features, loss, prop_entropy, prob_map], 
                    shot_capacity=hp['shot_capacity'], 
                    clip_weights=clip_weights, 
                    beta=hp['beta']
                )

                # C. Refinement
                final_logits = clip_logits.clone() / 100.0
                prob1 = torch.softmax(final_logits, dim=1)
                ent1 = softmax_entropy(prob1, False)

                if args.use_new_approximation:
                    final_logits2, new_info = compute_cache_logits(
                        pc_features, cache, clip_weights, 
                        [add_new_center, L_reg_old, L_reg_old_inv], 
                        i, hp
                    )
                    L_reg_old, L_reg_old_inv = new_info[0], new_info[1]
                else:
                    final_logits2 = compute_cache_logits_old(
                        pc_features, cache, clip_weights, hp
                    )

                # D. Combine Logits
                prob2 = torch.softmax(final_logits2, dim=1)
                ent2 = softmax_entropy(prob2, False)
                final_logits = ((1/ent1).reshape(-1,1) * prob1 + (1/ent2).reshape(-1,1) * prob2)


            end_event.record()
            torch.cuda.synchronize()
            stored_times.append(start_event.elapsed_time(end_event))

            (acc1, acc3, acc5), correct = accuracy(final_logits, target, topk=(1, 3, 5))
            acc1, acc3, acc5 = scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

    logging.info(f'Final Results: Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    total_time_ms = sum(stored_times)
    logging.info(f"Total time: {total_time_ms:.3f} ms")
    
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}