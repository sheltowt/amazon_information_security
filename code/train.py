from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier

def main():
  dataset = genfromtxt(open("/Users/hackreactor/code/sheltowt/amazon_information_security/data/train.csv", "r"), delimiter=',', dtype='f8')[1:]
  target = [x[0] for x in dataset]
  train = [x[1:] for x in dataset]
  test = genfromtxt(open("/Users/hackreactor/code/sheltowt/amazon_information_security/data/newtest.csv", "r"), delimiter=',', dtype='f8')[1:]

  rf = RandomForestClassifier(n_estimators = 100)
  rf.fit(train, target)

  test = [x[1:] for x in test]

  predicted_probs = [x[1] for x in rf.predict_proba(test)]

  savetxt('/Users/hackreactor/code/sheltowt/amazon_information_security/data/submission.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
  main()