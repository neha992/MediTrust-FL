from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from meditrust.federated.local_training import predict
from meditrust.config import TARGET
def evaluate(params, validation):
    y=validation[TARGET].to_numpy(); p=predict(params, validation)
    precision, recall, f1, _ = precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"precision":float(precision),"recall":float(recall),"f1":float(f1),"confusion_matrix":confusion_matrix(y,p).tolist()}
