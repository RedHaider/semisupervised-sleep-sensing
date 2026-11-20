from sleep_sensing.experiment import dataset,evaluate
def test_shape():
 x,y=dataset(20,2);assert x.shape==(20,7) and len(y)==20
def test_metrics():
 r=evaluate(3);assert 0<=r["pseudo_label_coverage"]<=1 and 0<=r["self_training"]["auroc"]<=1

