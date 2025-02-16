import torch
from tqdm import tqdm
import uuid

device = "cuda" if torch.cuda.is_available() else "mps"

def start(argv, model_name):
    num_data = int(argv[1])
    batch_size = int(argv[2])
    epoch = int(argv[3])
    uid = uuid.uuid4()
    model_path = f"{model_name}_{uid}"
    if int(len(argv)) == 5:
        model_path = argv[4]
    print("Number of data:", num_data)
    print("Batch size:", batch_size)
    print("Epoch:", epoch)
    print("Model path:", model_path)
    return num_data, batch_size, epoch, model_path


def main(train_dataloader, model, epoch, model_path):
    model = model().to(device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
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