Wijziging: MNIST → FashionMNIST
Vervang in het data-gedeelte:

python
Copy
Edit
trainset = torchvision.datasets.MNIST(...)
testset = torchvision.datasets.MNIST(...)
Met:

python
Copy
Edit
trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
✅ Bonus (optioneel): Labels voor confusion matrix verbeteren
De FashionMNIST dataset heeft andere labels dan cijfers (zoals "T-shirt", "Shoe", etc.). Voor een duidelijkere confusion matrix kun je deze labels tonen:

Voeg bovenin toe:

python
Copy
Edit
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
En verander de confusion matrix plot naar:

python
Copy
Edit
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
✅ Controleer: Output Directory
Zorg dat je script toestemming heeft om afbeeldingen op te slaan in de map waar je het uitvoert, of sla op in een submap zoals outputs/.

✅ Samenvatting van wijzigingen
Hier is het minimale dat je hoeft te wijzigen in je script:

1. Dataset:

python
Copy
Edit
trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
2. (Optioneel) Labels in confusion matrix:

python
Copy
Edit
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
...
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
🎯 Output
Na het draaien van je script:

learning_curves.png toont de verliescurves voor alle modellen

confusion_matrix_*.png toont de verwarringsmatrix per model

wrong_prediction_*.png toont 10 fout geclassificeerde voorbeelden per model

