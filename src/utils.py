import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    """Plot the feature importance of the trained model."""
    importances = model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
