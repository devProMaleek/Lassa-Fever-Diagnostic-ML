import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Binary Classification WebApp")    
    st.subheader("Are you a Lassa Fever Patient? 👋")

    st.sidebar.title("Binary Classification")
    st.sidebar.markdown("Are you a Lassa Fever Patient?")

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('Lassa Dataset.csv')
        label = LabelEncoder()

        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        
        # Feature Scaling.
        data=data.drop('Chronic Lung Disease',axis=1)
        data=data.drop('Heart Disease',axis=1)
        data=data.drop('Seizures',axis=1)
        data=data.drop('Blue Lips',axis=1)
        data=data.drop('Organ Failure',axis=1)
        data=data.drop('Hepatitis',axis=1)
        data=data.drop('Shock ',axis=1)
        
        return data

    @st.cache(persist = True)
    def split(df):
        x = df.drop('Lassa Fever',axis=1)
        y = df['Lassa Fever']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels = display_label)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            plt.plot([0, 1], [0, 1], linestyle = '--', color = 'red')
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Yes', 'No']
    display_label = ['Positive', 'Negative']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Random Forest", "Decision Tree", "KNeighbors"))       

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")        
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
        # max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"), key = 'criterion')
        # bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,  random_state=0)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:,1]
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("ROC AUC SCORE:", roc_auc_score(y_test, y_prob).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Accuracy in Percentage: ", (accuracy*100).round(2))       
            st.text('Classification Report:\n ' + classification_report(y_test, y_pred))
            plot_metrics(metrics)
            
    if classifier == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")
        criterion = st.sidebar.selectbox("Criterion", ("entropy", "gini", "log_loss"), key = 'criterion')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Decision Tree Results")  
            model = DecisionTreeClassifier(criterion = criterion,  random_state=0)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:,1]
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("ROC AUC SCORE:", roc_auc_score(y_test, y_prob).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Accuracy in Percentage: ", (accuracy*100).round(2))       
            st.text('Classification Report:\n ' + classification_report(y_test, y_pred))
            plot_metrics(metrics)
            
    if classifier == "KNeighbors":
        st.sidebar.subheader("Model Hyperparameters")
        n_neighbors = st.sidebar.number_input("The number of Neighbors", 5, 20, step = 1, key = 'n_neighbors')
        p = st.sidebar.number_input("The Power Parameter", 1, 10, step = 1, key = 'p')
        metric = st.sidebar.selectbox("Metric", ("minkowski", "euclidean"), key = 'metric')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("KNeighbors Results") 
            model = KNeighborsClassifier(n_neighbors = n_neighbors, p = p, metric = metric)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:,1]
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("ROC AUC SCORE:", roc_auc_score(y_test, y_prob).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Accuracy in Percentage: ", (accuracy*100).round(2))       
            st.text('Classification Report:\n ' + classification_report(y_test, y_pred))
            plot_metrics(metrics)

                        
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Lassa Fever Data Set (Classification)")
        st.write(df)
    
if __name__ == '__main__':
    main()