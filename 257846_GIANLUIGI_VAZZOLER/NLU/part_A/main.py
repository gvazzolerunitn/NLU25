# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import os
from functions import *
from utils import *
from model import *

if __name__ == "__main__":
    #Write the code to load the datasets and to run your functions
    # Print the results

    # Verifica se CUDA è disponibile
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Usa la GPU
        print(f"Stai usando la GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # Usa la CPU
        print("Stai usando la CPU")

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

# To save the model

path = 'model_bin/model_AdamW.pt'
path_best = 'model_bin/model_AdamW_best.pt'
os.makedirs(os.path.dirname(path), exist_ok=True)
torch.save(model.state_dict(), path)
torch.save(best_model.state_dict(), path_best)

# To visualize the model behavior

def plot_training_curves(sampled_epochs, losses_train, losses_dev, best_ppl):
    plt.figure(figsize=(12, 5))
    
    # Plot della loss
    plt.subplot(1, 2, 1)
    plt.plot(sampled_epochs, losses_train, label='Training Loss')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot della perplexity (opzionale - se hai perplexity salvata)
    plt.subplot(1, 2, 2)
    perplexity_dev = [np.exp(loss) for loss in losses_dev]
    plt.plot(sampled_epochs, perplexity_dev, label='Validation Perplexity', color='orange')
    plt.axhline(y=best_ppl, color='r', linestyle='--', label=f'Best Perplexity: {best_ppl:.2f}')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Salva il grafico
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_curves(sampled_epochs, losses_train, losses_dev, best_ppl)

    