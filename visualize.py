import matplotlib.pyplot as plt
import numpy as np


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
