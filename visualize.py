import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


def plot_feature_importances(model, feature_names, title="SVM_Feature_Importances"):
    clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
    
    if hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):   # add this branch
        importances = clf.feature_importances_
    elif hasattr(clf, 'theta_') and hasattr(clf, 'var_'):
        importances = (clf.theta_[1] - clf.theta_[0]) / np.sqrt(clf.var_[1] + clf.var_[0])
    else:
        print(f"[yellow]Model {title} does not have coefficients or means (not a supported model).[/yellow]")
        return
        
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    
    colors = ['salmon' if val < 0 else 'mediumseagreen' for val in importances[indices]]
    
    plt.barh(range(len(indices)), importances[indices], color=colors, align='center')
    safe_feature_names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices]
    plt.yticks(range(len(indices)), safe_feature_names)
    plt.xlabel('Coefficient / Effect Value (Importance)')
    plt.tight_layout()
    
    filename = f"graphs/{title.lower().replace(' ', '_')}_importance.png"
    plt.savefig(filename)
    
    print(f"[dim]Saved feature importances to {filename}[/dim]")
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, title, output_dir="graphs"):
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'Machine'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f"Confusion Matrix: {title}")
    
    clean_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{output_dir}/{clean_title}_confusion.png"
    plt.savefig(filename)
    
    from rich import print
    print(f"[dim]Saved Confusion Matrix to {filename}[/dim]")
    plt.close()

def plot_mlm_analysis(human_probs, human_ranks, machine_probs, machine_ranks, split_name="Dataset", output_dir="graphs"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Probabilities
    ax1.boxplot([human_probs, machine_probs], labels=['Human', 'Machine'])
    ax1.set_title(f"MLM Target Probability ({split_name})")
    ax1.set_ylabel("Probability")
    
    # Ranks Boxplot
    ax2.boxplot([human_ranks, machine_ranks], labels=['Human', 'Machine'])
    ax2.set_title(f"MLM Target Rank ({split_name})")
    ax2.set_ylabel("Rank")
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    clean_title = split_name.lower().replace(" ", "_")
    filename = f"{output_dir}/mlm_analysis_{clean_title}.png"
    plt.savefig(filename)
    
    from rich import print
    print(f"[dim]Saved MLM Analysis Plot to {filename}[/dim]")
    plt.close()
