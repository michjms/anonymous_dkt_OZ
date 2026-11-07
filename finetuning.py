device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_criterion = nn.BCELoss(reduction='none')

fractions = [0.1]
num_epochs = 50
patience = 3
results = {}

n_target = len(train_dataset_target)
indices = torch.randperm(n_target, generator=torch.Generator().manual_seed(42))
val_size = int(0.125 * n_target)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

ft_val_subset = Subset(train_dataset_target, val_indices)
ft_val_loader = DataLoader(ft_val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

for frac in fractions:
    print(f"\n{'='*50}")
    print(f"Target fraction: {frac}")
    print(f"{'='*50}")

    model = DKTModel(num_skills).to(device)
    for param in model.parameters():
        param.requires_grad = True

    ft_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    # Subsample from train portion only (not val)
    subset_size = int(frac * len(train_indices))
    perm = torch.randperm(len(train_indices))[:subset_size]
    ft_train_subset = Subset(train_dataset_target, train_indices[perm])
    ft_loader = DataLoader(ft_train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0, 0

        for inputs, targets, skills, lengths in ft_loader:
            inputs, targets, skills, lengths = (
                inputs.to(device), targets.to(device), skills.to(device), lengths.to(device)
            )
            ft_optimizer.zero_grad()
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

            loss = label_criterion(outputs, targets)
            loss = (loss * mask.float()).sum() / mask.sum()
            loss.backward()
            ft_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        val_auc = eval_auc(model, ft_val_loader)
        test_auc = eval_auc(model, target_test_loader)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model and evaluate on test
    model.load_state_dict(best_model_state)
    final_test_auc = eval_auc(model, target_test_loader)
    print(f"Best Val AUC: {best_val_auc:.4f} | Final Test AUC: {final_test_auc:.4f}")
    results[frac] = final_test_auc