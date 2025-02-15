import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps"

def main(train_dataloader, model, epoch, model_path):
    model = model().to(device)
    loss_function = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in tqdm(range(epoch)):
        for sn_time, signal_time in train_dataloader:
            optimiser.zero_grad()
            outputs = model(sn_time.to(device))
            loss = loss_function(outputs, signal_time.to(device))
            loss.backward()
            optimiser.step()

    torch.save(model.state_dict(), f"{model_path}.pth")
    print("Model saved to", f"{model_path}.pth")