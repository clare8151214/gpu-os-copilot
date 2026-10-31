"""Actual image-classifier forward/backward steps on fixed synthetic inputs."""


def build_operation(torch, *, training: bool):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 10),
    ).cuda()
    model.train(training)
    inputs = torch.randn(32, 3, 32, 32, device="cuda")
    targets = torch.randint(0, 10, (32,), device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if training else None
    initial = next(model.parameters()).detach().clone()
    latest = {}

    def operation():
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            latest["loss"] = loss.detach()
        else:
            with torch.inference_mode():
                logits = model(inputs)
        latest["logits"] = logits.detach()

    def validate():
        # Outside the measured NVTX/timing region. Reject NaNs and no-op training.
        logits = latest["logits"]
        finite = bool(torch.isfinite(logits).all().item()) and all(
            bool(torch.isfinite(p).all().item()) for p in model.parameters()
        )
        changed = not torch.equal(initial, next(model.parameters()).detach())
        loss = float(latest["loss"].item()) if training else None
        valid = finite and (changed if training else not changed)
        if not valid:
            raise RuntimeError("Model output/parameter validation failed")
        return {
            "validated": True,
            "finite": finite,
            "parameters_changed": changed,
            "output_shape": list(logits.shape),
            "output_sum": float(logits.sum().item()),
            "final_loss": loss,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    return operation, validate
