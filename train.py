mean_correct = [seq[:, 1].mean() for seq in sequencesNOZ_final]
bins = pd.qcut(mean_correct, q=4, labels=False, duplicates='drop')
train_seq, test_seq = train_test_split(sequencesNOZ_final, test_size=0.2, random_state=42, stratify=bins)

mean_correct_train = [seq[:, 1].mean() for seq in train_seq]
bins_train = pd.qcut(mean_correct_train, q=4, labels=False, duplicates='drop')
train_seq, val_seq = train_test_split(train_seq, test_size=0.125, random_state=42, stratify=bins_train)

train_seq = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in train_seq]
val_seq = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in val_seq]
test_seq = [truncate_sequence(seq, MAX_SEQ_LEN) for seq in test_seq]

train_dataset = DKTDataset(train_seq, num_skills)
val_dataset = DKTDataset(val_seq, num_skills)
test_dataset = DKTDataset(test_seq, num_skills)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DKTModel(num_skills).to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50 
patience = 3
best_val_auc = 0.0
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for inputs, targets, skills, lengths in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        skills = skills.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)

        batch_size, max_len, _ = outputs.shape
        length_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        length_mask = length_mask.unsqueeze(2).expand(-1, -1, num_skills)

        skill_mask = torch.zeros(batch_size, max_len, num_skills, device=device)
        valid_indices = (skills >= 0) & (skills < num_skills)

        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len)[valid_indices]
        time_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)[valid_indices]
        skill_indices = skills[valid_indices]

        skill_mask[batch_indices, time_indices, skill_indices] = 1
        mask = length_mask & (skill_mask == 1)

        loss = criterion(outputs, targets)
        loss = (loss * mask.float()).sum() / mask.sum()

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for inputs, targets, skills, lengths in val_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            outputs = model(inputs, lengths)
            for b in range(len(lengths)):
                for t in range(lengths[b]):
                    skill_id = skills[b, t].item()
                    if skill_id >= 0:
                        val_preds.append(outputs[b, t, skill_id].item())
                        val_targets.append(targets[b, t, skill_id].item())

    val_auc = roc_auc_score(val_targets, val_preds)
    print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break