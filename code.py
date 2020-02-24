import pandas as pd 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class predictor():
    
    def __init__(self):
        self.list_of_classifiers = [LogisticRegression(solver='lbfgs'), 
                                   KNeighborsClassifier(),
                       SVC(probability=True), DecisionTreeClassifier(), 
                       GaussianNB(), 
                       Perceptron(max_iter=40, eta0=0.1, random_state=0),
                       BernoulliNB(), 
                       DecisionTreeClassifier(), 
                       AdaBoostClassifier(),RandomForestClassifier(),
                       BaggingClassifier(), MLPClassifier(),
                       GradientBoostingClassifier(), ExtraTreesClassifier(),
                       BernoulliNB(), RidgeClassifier(), Lasso()     
                       ]
        self.path = r'C:\Users\WINDOWS\Desktop\HobbyProjects\bank.csv'
        self.cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome','y']
           
    def execute(self):
        choosen_classifier = self.please_select_a_classifier()
        self.predict_the_output(choosen_classifier)
        
    def predict_the_output(self, choosen_classifier):
        fdata = self.do_categorical_encoding()
        ncols = fdata.shape[1]

        #spilitting into x and y
        x = fdata.iloc[:,0:ncols-1]
        y = fdata.iloc[:,ncols-1:ncols]
        
        train_x,test_x,train_y,test_y = self.please_split_the_data_into_test_and_train(x,y)
 
        choosen_classifier.fit(train_x,train_y)
        prediction = choosen_classifier.predict(test_x)
        self.generate_report(prediction, test_y, choosen_classifier, test_x)
       
    def generate_report(self, prediction, test_y, choosen_classifier, test_x):
        print("Accuracy score = ",accuracy_score(prediction, test_y))
        print("Confusion matrix \n",confusion_matrix(prediction, test_y))
        print("Classification report \n",classification_report(prediction, test_y))

        probs = choosen_classifier.predict_proba(test_x)
        probs = probs[:, 1]
        auc = roc_auc_score(test_y, probs)
        print("AUC : ",auc)
        print("\n --------------------------- \n")
    
    def please_split_the_data_into_test_and_train(self,x,y):
        #splitting into train and test with training -75% and test -25%
        train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)
        
             #Scaling all data; mandatory
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        
        PCA_toggle = input("Do you wish to do PCA? \n Press Y or N to record your answer")
        if PCA_toggle == "Y":
            numColsIwant = int(input("Enter the number of features that you want"))
            train_x,test_x = self.doPcaOnThisPlease(numColsIwant,train_x,test_x)
        elif PCA_toggle == "N":
            pass
        else:
            print("Invalid response")
        
        return train_x,test_x,train_y,test_y
      
    def please_select_a_classifier(self):
        for i in self.list_of_classifiers:
            print("press", self.list_of_classifiers.index(i), "for ", i )
        choosen_classifier = self.list_of_classifiers[int(input("enter your choice"))]
        print("Classifying the data")    
        return choosen_classifier
        
    def do_categorical_encoding(self):
        print("Loading the data ...")
        #fdata has full data
        fdata = pd.read_csv(self.path)

        print("Loading done. Processing ...")
        #Categorical cols which have text
        oe = ce.OrdinalEncoder(cols = self.cat_cols)
        fdata = oe.fit_transform(fdata)
        return fdata
       
    def doPcaOnThisPlease(self, reduced_comps_that_I_want,train_x,test_x):
        print("I am doing PCA now...")
        pca = PCA(n_components=reduced_comps_that_I_want)
        pca.fit(train_x)
        reduced_cols = pca.n_components_
        print("# of PCA components: ",reduced_cols)
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)
        return train_x,test_x
    
Predictor = predictor()
Predictor.execute()
