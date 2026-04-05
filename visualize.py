import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances(model, feature_names, title="SVM_Feature_Importances"):
    clf = model.named_steps['clf']
    
    if hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        
        colors = ['red' if val < 0 else 'green' for val in importances[indices]]
        
        plt.barh(range(len(indices)), importances[indices], color=colors, align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Coefficient Value (Importance)')
        plt.tight_layout()
        
        filename = f"output/{title.lower().replace(' ', '_')}_importance.png"
        plt.savefig(filename)
        
        print(f"[dim]Saved feature importances to {filename}[/dim]")
        plt.close()
    else:
        print(f"[yellow]Model {title} does not have coefficients (not a linear model).[/yellow]")
