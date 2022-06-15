import numpy as np
import streamlit as st
import pickle

# Loading the saved Model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

st.set_page_config(page_title="Lassa Diagnostic App")

def testPage():
    import streamlit as st
    st.sidebar.success("Select a Page above.")

    # Creating a function for Prediction
    def lassa_fever_prediction(input_data):
        # Changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            return 'You are not diagnosed with Lassa Fever'
        else:
            return 'You are diagnosed with Lassa Fever'

    # Convert User report 
    def convert_userInput_into_number(fieldName):
        if fieldName == "Yes":
            fieldName = 1
        else:
            fieldName = 0
        return fieldName

    def main():
        # Page Title
        st.header('Lassa Fever Diagnostic System 👋')

        # Getting the input from the user
        
        col1, col2 = st.columns(2)
        
        with col1:
            breathing_problem = st.radio(
                "Are you experiencing Breathing Problem?",
                ('Yes', 'No'), horizontal=True,)
            breathing_problem = convert_userInput_into_number(breathing_problem)
            
            fever = st.radio(
                "Are you experiencing Serious Fever?",
                ('Yes', 'No'), horizontal=True)
            fever = convert_userInput_into_number(fever)

            dry_cough = st.radio(
                "Do you have Dry Cough?",
                ('Yes', 'No'), horizontal=True)
            dry_cough = convert_userInput_into_number(dry_cough)

            sore_throat = st.radio(
                "Do you have Sore Throat?",
                ('Yes', 'No'), horizontal=True)
            sore_throat = convert_userInput_into_number(sore_throat)

            running_nose = st.radio(
                "Do you have Runny Nose?",
                ('Yes', 'No'), horizontal=True)
            running_nose = convert_userInput_into_number(running_nose)

            chest_pain = st.radio(
                "Do you have Chest Pain?",
                ('Yes', 'No'), horizontal=True)
            chest_pain = convert_userInput_into_number(chest_pain)
            
            headache = st.radio(
                "Are you experiencing Headache?",
                ('Yes', 'No'), horizontal=True)
            headache = convert_userInput_into_number(headache)
            
        with col2:

            Diabetes = st.radio(
                "Are you Diabetic?",
                ('Yes', 'No'), horizontal=True)
            Diabetes = convert_userInput_into_number(Diabetes)

            hypertension = st.radio(
                "Are you hypertensive?",
                ('Yes', 'No'), horizontal=True)
            hypertension = convert_userInput_into_number(hypertension)

            fatigue = st.radio(
                "Are you experiencing Fatigue(Weakness)?",
                ('Yes', 'No'), horizontal=True)
            fatigue = convert_userInput_into_number(fatigue)

            diarrhoea = st.radio(
                "Do you have Diarrhoea?",
                ('Yes', 'No'), horizontal=True)
            diarrhoea = convert_userInput_into_number(diarrhoea)

            vomiting = st.radio(
                "Are you Vomiting?",
                ('Yes', 'No'), horizontal=True)
            vomiting = convert_userInput_into_number(vomiting)

            hearing_loss = st.radio(
                "Do you have Hearing Problem?",
                ('Yes', 'No'), horizontal=True  )
            hearing_loss = convert_userInput_into_number(hearing_loss)

        # All User Input
        user_input = [breathing_problem, fever, dry_cough, sore_throat, running_nose, chest_pain,
                    headache, Diabetes, hypertension, fatigue, diarrhoea, vomiting, hearing_loss]
    
        # Code for prediction
        diagnosis = ''

        # Prediction Button
        btn = st.button("Lassa Fever Test Result")

        if btn:
            diagnosis = lassa_fever_prediction(user_input)
        
        # Printing the Results
        st.success(diagnosis)
    if __name__ == '__main__':
        main() 

def Visualization():
    import streamlit as st
    import pandas as pd
    import numpy as np

    # Data Visualization Library
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Page Title
    st.title('Visualization Page 👋')
    data = pd.read_csv('Lassa Dataset.csv')
    st.sidebar.subheader("Choose Classifier")
    visualization = st.sidebar.selectbox("Charts", ("Count Of Cases", "Breathing Problem", "Sore throat", "Fever", "Chest Pain", "Diabetes", "Vomitting", "Fatigue", "Headache", "Shock ", "Organ Failure", "Hepatitis", "Seizures", "Chronic Lung Disease")) 
    
    if visualization == "Count Of Cases":
        col1, col2 = st.columns(2)
        with col1:
            sns.countplot(x='Lassa Fever',data=data, palette=['#ff1ac6',"#FAAE7B"])
            st.pyplot()
            st.write("Count of Lassa Fever")
        with col2: 
            data["Lassa Fever"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
            plt.title('Count of Cases');
            st.pyplot()
    
    def Column_Analysis(column_name, palette): 
        if visualization == column_name:
            col1, col2 = st.columns(2)
            with col1:
                sns.countplot(x=column_name,hue='Lassa Fever',data=data, palette=palette)
                st.pyplot()
            with col2:
                data[column_name].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
                plt.title(column_name);
                st.pyplot()
                
    for col in data.columns:
        Column_Analysis(col, ['#ff1ac6',"#FAAE7B"])

        
def topThreeModels():
    import streamlit as st
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
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
    
    


page_names_to_funcs = {
    "Test Page": testPage,
    "Visualization": Visualization,
    "Top Three Models": topThreeModels,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
        
