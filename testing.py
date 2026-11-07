from sklearn.utils import resample

def bootstrap_metric(all_preds, all_targets, student_indices, metric_fn=roc_auc_score, n_bootstrap=1000, ci=95):
    scores = []
    for _ in range(n_bootstrap):
        sampled = resample(student_indices, random_state=None)
        boot_preds, boot_targets = [], []
        for start, end in sampled:
            boot_preds.extend(all_preds[start:end])
            boot_targets.extend(all_targets[start:end])
        try:
            scores.append(metric_fn(np.array(boot_targets), np.array(boot_preds)))
        except ValueError:
            continue
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), lower, upper

def eval_with_bootstrap(model_to_eval, loader, device_to_use, num_skills, label="", n_bootstrap=1000):
    model_to_eval.eval()
    all_preds = []
    all_targets = []
    student_indices = []
    with torch.no_grad():
        for inputs, targets, skills, lengths in loader:
            inputs = inputs.to(device_to_use)
            targets = targets.to(device_to_use)
            skills = skills.to(device_to_use)
            lengths = lengths.to(device_to_use)
            outputs = model_to_eval(inputs, lengths)
            for b in range(len(lengths)):
                start = len(all_preds)
                for t in range(lengths[b]):
                    skill_id = skills[b, t].item()
                    if skill_id >= 0:
                        all_preds.append(outputs[b, t, skill_id].item())
                        all_targets.append(targets[b, t, skill_id].item())
                end = len(all_preds)
                if end > start:
                    student_indices.append((start, end))
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    auc_mean, ci_low, ci_high = bootstrap_metric(all_preds, all_targets, student_indices, roc_auc_score, n_bootstrap)
    if label:
        print(f'{label} AUC: {auc:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]')
    return auc, ci_low, ci_high, all_preds, all_targets, student_indices

auc, ci_low, ci_high, preds_ss, targets_ss, si_ss = eval_with_bootstrap(
    model, test_loader, device, num_skills)

auc, ci_low, ci_high, preds_st, targets_st, si_st = eval_with_bootstrap(
    model, target_test_loader, device, num_skills)

