patience = 7
monitor = 'source'  
dann_checkpoint_path = '[removed identifying information]'

def split_val(sequences, val_size=0.125, random_state=42):
    mean_correct = [seq[:, 1].mean() for seq in sequences]
    bins = pd.qcut(mean_correct, q=4, labels=False, duplicates='drop')
    return train_test_split(sequences, test_size=val_size, random_state=random_state, stratify=bins)

def dann_auc(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets, skills, lengths in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            skills = skills.to(device)
            outputs = model(inputs, lengths) 

            batch_size, max_len, _ = outputs.shape
            length_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            length_mask = length_mask.unsqueeze(2).expand(-1, -1, num_skills)
            skill_mask = torch.zeros(batch_size, max_len, num_skills, device=device)
            valid_indices = (skills >= 0) & (skills < num_skills)
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len)[valid_indices]
            time_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)[valid_indices]
            skill_idx = skills[valid_indices]
            skill_mask[batch_idx, time_idx, skill_idx] = 1
            mask = length_mask & (skill_mask == 1)

            preds.append(outputs.cpu()[mask.cpu()])
            labels.append(targets[mask.cpu()])

    return roc_auc_score(torch.cat(labels).numpy(), torch.cat(preds).numpy())

train_seq_source, test_seq_source = splitting(sequencesNOZ_final)
train_seq_source, val_seq_source = split_val(train_seq_source)
train_seq_source = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in train_seq_source]
val_seq_source = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in val_seq_source]
test_seq_source = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in test_seq_source]

train_dataset_source = DKTDataset(train_seq_source, num_skills)
val_dataset_source = DKTDataset(val_seq_source, num_skills)
test_dataset_source = DKTDataset(test_seq_source, num_skills)

source_train_loader = DataLoader(train_dataset_source, batch_size=32, shuffle=True, collate_fn=collate_fn)
source_val_loader = DataLoader(val_dataset_source, batch_size=32, shuffle=False, collate_fn=collate_fn)
source_test_loader = DataLoader(test_dataset_source, batch_size=32, shuffle=False, collate_fn=collate_fn)

train_seq_target, test_seq_target = splitting(sequences10_final)
train_seq_target, val_seq_target = split_val(train_seq_target)
train_seq_target = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in train_seq_target]
val_seq_target = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in val_seq_target]
test_seq_target = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in test_seq_target]

train_dataset_target = DKTDataset(train_seq_target, num_skills)
val_dataset_target = DKTDataset(val_seq_target, num_skills)
test_dataset_target = DKTDataset(test_seq_target, num_skills)

target_train_loader = DataLoader(train_dataset_target, batch_size=32, shuffle=True, collate_fn=collate_fn)
target_val_loader = DataLoader(val_dataset_target, batch_size=32, shuffle=False, collate_fn=collate_fn)
target_test_loader = DataLoader(test_dataset_target, batch_size=32, shuffle=False, collate_fn=collate_fn)

best_val_auc = 0.0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_label_loss = 0
    train_domain_loss = 0

    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    num_batches = max(len(source_train_loader), len(target_train_loader))

    p = epoch / num_epochs
    # lambda_ = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
    lambda_ = 0.01 + 0.04 * (2 / (1 + np.exp(-10 * p)) - 1)  # caps around 0.05
    model.grl.set_lambda(lambda_)

    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{num_epochs}')

    for _ in pbar:
        try:
            source_inputs, source_targets, source_skills, source_lengths = next(source_iter)
        except StopIteration:
            source_iter = iter(source_train_loader)
            source_inputs, source_targets, source_skills, source_lengths = next(source_iter)

        try:
            target_inputs, _, target_skills, target_lengths = next(target_iter)
        except StopIteration:
            target_iter = iter(target_train_loader)
            target_inputs, _, target_skills, target_lengths = next(target_iter)

        source_inputs = source_inputs.to(device)
        source_targets = source_targets.to(device)
        source_skills = source_skills.to(device)
        source_lengths = source_lengths.to(device)

        target_inputs = target_inputs.to(device)
        target_skills = target_skills.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        source_outputs, source_domain_pred = model(source_inputs, source_lengths, return_domain=True)

        batch_size, max_len, _ = source_outputs.shape
        length_mask = torch.arange(max_len, device=device).unsqueeze(0) < source_lengths.unsqueeze(1)
        length_mask = length_mask.unsqueeze(2).expand(-1, -1, num_skills)

        skill_mask = torch.zeros(batch_size, max_len, num_skills, device=device)
        valid_indices = (source_skills >= 0) & (source_skills < num_skills)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len)[valid_indices]
        time_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)[valid_indices]
        skill_indices = source_skills[valid_indices]
        skill_mask[batch_indices, time_indices, skill_indices] = 1

        mask = length_mask & (skill_mask == 1)
        label_loss = label_criterion(source_outputs, source_targets)
        label_loss = (label_loss * mask.float()).sum() / mask.sum()

        _, target_domain_pred = model(target_inputs, target_lengths, return_domain=True)

        source_domain_labels = torch.zeros(source_inputs.size(0), dtype=torch.long, device=device)
        target_domain_labels = torch.ones(target_inputs.size(0), dtype=torch.long, device=device)

        all_domain_preds = torch.cat([source_domain_pred, target_domain_pred])
        all_domain_labels = torch.cat([source_domain_labels, target_domain_labels])

        domain_loss = domain_criterion(all_domain_preds, all_domain_labels)
        with torch.no_grad():
            domain_pred_labels = torch.argmax(all_domain_preds, dim=1)
            domain_acc = (domain_pred_labels == all_domain_labels).float().mean().item()

        # === Total loss ===
        total_loss = label_loss + lambda_*domain_loss

        total_loss.backward()
        optimizer.step()

        train_label_loss += label_loss.item()
        train_domain_loss += domain_loss.item()
        pbar.set_postfix({'label_loss': label_loss.item(), 'domain_loss': domain_loss.item()})

    print(f'Epoch {epoch+1}/{num_epochs}, Label Loss: {train_label_loss/num_batches:.4f}, Domain Loss: {train_domain_loss/num_batches:.4f}, Domain Acc: {domain_acc:.4f}')

    src_val_auc = dann_auc(model, source_val_loader)
    tgt_val_auc = dann_auc(model, target_val_loader)

    val_auc = src_val_auc if monitor == 'source' else tgt_val_auc
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

checkpoint = torch.load(dann_checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

src_test_auc = dann_auc(model, source_test_loader)
tgt_test_auc = dann_auc(model, target_test_loader)