import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix , mean_squared_error, mean_absolute_error, r2_score , silhouette_score , precision_score , recall_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering , DBSCAN , AffinityPropagation
from sklearn.svm import SVC
import os 
import pickle 

# placement_data = pd.read_csv("/content/PlacementPredictiondata.csv")

placement_data = pd.read_csv("/workspaces/PLACEMENT-PREDICTION-APP/PlacementPredictiondata.csv")

# placement_data = pd.read_csv("PlacementPredictiondata.csv")

label_encoder = LabelEncoder()
placement_data['PlacementStatus'] = label_encoder.fit_transform(placement_data['PlacementStatus'])

placement_data['ExtracurricularActivities'] = placement_data['ExtracurricularActivities'].map({'No': 0,'Yes' : 1})
placement_data['PlacementTraining'] = placement_data['PlacementTraining'].map({'No': 0,'Yes' : 1})

X = placement_data[['CGPA','Internships','Projects', 'AptitudeTestScore', 'SoftSkillsRating',
                          'Workshops/Certifications','SSC_Marks','HSC_Marks','ExtracurricularActivities','PlacementTraining']]
y = placement_data["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

st.subheader("PLACEMENT PREDICTION ANALYSIS")
st.write(placement_data.head())


models = { 
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC() ,
    #"Agglomerative Clustering": AgglomerativeClustering(),
    #"DBSCAN": DBSCAN(),
    #"Affinity Propagation": AffinityPropagation()
}

if not os.path.exists("models"):
    os.makedirs("models")

for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'models/{name}.pkl', 'wb') as file:
        pickle.dump(model, file)


option = st.radio("Select an Option:", [
    "Evaluate a Model", 
    "Compare All Models",
    "Make a Prediction"
])

if option == "Evaluate a Model":
    st.subheader("Select a Model to Evaluate")
    if st.button("Decision Tree Classifier "):
        placement_data_model = DecisionTreeClassifier()
        placement_data_model.fit(X_train,y_train)
        ans_placement = placement_data_model.predict(X_train)
        accuracy = accuracy_score(y_train , ans_placement) * 100
        st.write(f"Accuracy: {accuracy:.2f}%")
        #mse = mean_squared_error(y_test, ans_placement)
        #mae = mean_absolute_error(y_test, ans_placement)
        #r2 = r2_score(y_test, ans_placement)
        #st.write(f"Mean Squared Error: {mse}")
        #st.write(f"Mean Absolute Error: {mae}")
        #st.write(f"R² Score: {r2}")
        #report = classification_report(y_test, ans_placement)
        #formatted_report = report.replace(" ","  ")
        #st.dataframe(report)
        #st.write(formatted_report)
        #matrix = confusion_matrix(y_test, ans_placement)
        #st.write(matrix)


        # ACCURACY FOR TRAINING SET 
        st.write("TRAIN SET")
        accuracy = accuracy_score(y_train, ans_placement)*100
        precision = precision_score(y_train, ans_placement, average='weighted')*100
        recall = recall_score(y_train , ans_placement , average='weighted')*100
        report = classification_report(y_train , ans_placement)
        cm = confusion_matrix(y_train , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)


        # ACCURACY FOFR TESTING SET 
        st.write("TEST SET ")
        ans_placement = placement_data_model.predict(X_test)
        accuracy = accuracy_score(y_test, ans_placement)*100
        precision = precision_score(y_test, ans_placement, average='weighted')*100
        recall = recall_score(y_test , ans_placement , average='weighted')*100
        report = classification_report(y_test , ans_placement)
        cm = confusion_matrix(y_test , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))


    if st.button("KNN CLASSIFIER"):
        placement_data_model = KNeighborsClassifier()
        placement_data_model.fit(X_train , y_train)
        ans_placement = placement_data_model.predict(X_train)
        accuracy = accuracy_score(y_train , ans_placement)*100
        st.write(f"Accuracy: {accuracy:.2f}%")
        #mse = mean_squared_error(y_test , ans_placement)
        #mae = mean_absolute_error(y_test , ans_placement)
        #r2 = r2_score(y_test , ans_placement)
        #st.write(f"Mean Squared Error: {mse}")
        #st.write(f"Mean Absolute Error: {mae}")
        #st.write(f"R² Score: {r2}")
        #report = classification_report(y_test , ans_placement)
        #formatted_report = report.replace(" ","  ")
        #st.dataframe(report)
        #st.write(formatted_report)
        #matrix = confusion_matrix(y_test , ans_placement)
        #st.write(matrix)


        # ACCURACY FOR TRAINING SET
        st.write("TRAIN SET")
        accuracy = accuracy_score(y_train, ans_placement)*100
        precision = precision_score(y_train, ans_placement, average='weighted')*100
        recall = recall_score(y_train , ans_placement , average='weighted')*100
        report = classification_report(y_train , ans_placement)
        cm = confusion_matrix(y_train , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)

        # ACCURACY FOR TESTING SET
        st.write("TEST SET ")
        ans_placement = placement_data_model.predict(X_test)
        accuracy = accuracy_score(y_test, ans_placement)*100
        precision = precision_score(y_test, ans_placement, average='weighted')*100
        recall = recall_score(y_test , ans_placement , average='weighted')*100
        report = classification_report(y_test , ans_placement)
        cm = confusion_matrix(y_test , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
        #st.pyplot(fig)

      

    if st.button("SVM CLASSIFIER"):  
        placement_data_model = SVC()
        placement_data_model.fit(X_train , y_train)
        ans_placement = placement_data_model.predict(X_train)
        accuracy = accuracy_score(y_train , ans_placement)*100
        st.write(f"Accuracy: {accuracy:.2f}%")
        #mse = mean_squared_error(y_test , ans_placement)
        #mae = mean_absolute_error(y_test , ans_placement)
        #r2 = r2_score(y_test , ans_placement)
        #st.write(f"Mean Squared Error: {mse}")
        #st.write(f"Mean Absolute Error: {mae}")
        #st.write(f"R² Score: {r2}")
        #report = classification_report(y_test , ans_placement)
        #formatted_report = report.replace(" ","  ")
        #st.dataframe(report)
        #st.write(formatted_report) 
        #matrix = confusion_matrix(y_test , ans_placement)
        #st.write(matrix) 


    #selected_model = st.selectbox("Choose a model", list(model))
    #selected_model = st.selectbox("Choose a model", list(model.keys()))
    #with open(f'models/{selected_model}.pkl', 'rb') as file:
        #model = pickle.load(file)

    #y_pred = model.predict(X_test)
        
        # ACCURACY FOR TRAINING SET
        st.write("TRAIN SET")
        accuracy = accuracy_score(y_train, ans_placement)*100
        precision = precision_score(y_train, ans_placement, average='weighted')*100
        recall = recall_score(y_train , ans_placement , average='weighted')*100
        report = classification_report(y_train , ans_placement)
        cm = confusion_matrix(y_train , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)


        # ACCURACY FOR TEST SET
        st.write("TEST SET ")
        ans_placement = placement_data_model.predict(X_test)
        accuracy = accuracy_score(y_test, ans_placement)*100
        precision = precision_score(y_test, ans_placement, average='weighted')*100
        recall = recall_score(y_test , ans_placement , average='weighted')*100
        report = classification_report(y_test , ans_placement)
        cm = confusion_matrix(y_test , ans_placement)

        st.write(" Accuracy:", accuracy)
        st.write(" Precision:", precision)
        st.write(" Recall:", recall)
        st.text(" Classification Report:")
        st.text(report)
        st.text(" Confusion Matrix:")
        st.write(cm)

        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
        #st.pyplot(fig)


        # COMPARISON OF ALL MODELS 

elif option == "Compare All Models":
    st.subheader("Comparison of Model Performances")
    results = {}

    # TRAINING SET 

    st.write("TRAINING SET ")
    for name, model in models.items():
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train , y_pred)*100
        precision = precision_score(y_train , y_pred, average='weighted')*100
        recall = recall_score(y_train , y_pred, average='weighted')*100
        results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}

    df_metrics = pd.DataFrame(results).T
    st.dataframe(df_metrics)

    fig, ax = plt.subplots()
    df_metrics.plot(kind='bar', ax=ax)
    st.pyplot(fig)


    # TESTING SET 

    st.write("TEST SET ")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test , y_pred)*100
        precision = precision_score(y_test , y_pred, average='weighted')*100
        recall = recall_score(y_test , y_pred, average='weighted')*100
        results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall}

    df_metrics = pd.DataFrame(results).T
    st.dataframe(df_metrics)

    fig, ax = plt.subplots()
    df_metrics.plot(kind='bar', ax=ax)
    st.pyplot(fig)



elif option == "Make a Prediction":

    #st.subheader("Predict with Selected Model")
    #selected_model = st.selectbox("Choose a model for prediction", list(models.keys()))
    #if st.button("Decision Tree Classifier "):
    placement_data_model = DecisionTreeClassifier()
    placement_data_model.fit(X_train,y_train)
    ans_placement = placement_data_model.predict(X_test)
    cgpa = st.number_input("Enter CGPA   [ 0.00  to  10.00 ] ")
    internship = st.number_input("Enter Internship   [ 0 or 1 or 2 ] ")
    projects = st.number_input("Enter Projects   [ 0 or 1 or 2 ]")
    aptitude_score = st.number_input("Enter Aptitude Score   [ 0  to  100 ]")
    softskills_score = st.number_input("Enter Soft Skills Score   [ 0.0  to  5.00 ]")
    workshops = st.number_input("Enter Workshops    [ 0 or 1 or 2 ]")
    ssc_marks = st.number_input("Enter SSC Marks    [ 0  to  100 ]")
    hsc_marks = st.number_input("Enter HSC Marks    [ 0  to  100 ]")
    #st.write("NO : 0 and YES : 1")
    extracurricular_activities = st.number_input("Enter Extracurricular Activities  [ NO : 0  &  YES : 1 ]")
    #st.write("NO : 0 and YES : 1")
    placement_training = st.number_input("Enter Placement Training   [ NO : 0  &  YES : 1 ]")
    if st.button("PREDICT"):
        user_input = np.array([[cgpa,internship,projects,aptitude_score,softskills_score,workshops,ssc_marks,
                                    hsc_marks,extracurricular_activities,placement_training]])
        prediction = placement_data_model.predict(user_input)
        if prediction[0] == 0:
            st.write("Predicted Class: Not Placed")
        else:
            st.write("Predicted Class: Placed") 

    #if st.button("KNN CLASSIFIER"):
        #placement_data_model = KNeighborsClassifier()
        #placement_data_model.fit(X_train , y_train)
        #ans_placement = placement_data_model.predict(X_test)
        #cgpa = st.number_input("Enter CGPA")
        #internship = st.number_input("Enter Internship")
        #projects = st.number_input("Enter Projects")
        #aptitude_score = st.number_input("Enter Aptitude Score")
        #softskills_score = st.number_input("Enter Soft Skills Score")
        #workshops = st.number_input("Enter Workshops")
        #ssc_marks = st.number_input("Enter SSC Marks")
        #hsc_marks = st.number_input("Enter HSC Marks")
            #st.write("NO : 0 and YES : 1")
        #extracurricular_activities = st.number_input("Enter Extracurricular Activities [ NO : 0  &  YES : 1 ]")
            #st.write("NO : 0 and YES : 1")
       #placement_training = st.number_input("Enter Placement Training [ NO : 0  &  YES : 1 ]")
        #if st.button("PREDICT"):

            #user_input = np.array([[cgpa,internship,projects,aptitude_score,softskills_score,workshops,ssc_marks,
                                        #hsc_marks,extracurricular_activities,placement_training]])
            #prediction = placement_data_model.predict(user_input)
            #if prediction[0] == 0:
                #st.write("Predicted Class: Not Placed")
            #else:
                #st.write("Predicted Class: Placed") 


    #if st.button("SVM CLASSIFIER"):  
        #placement_data_model = SVC()
        #placement_data_model.fit(X_train , y_train)
        #ans_placement = placement_data_model.predict(X_test)
        #cgpa = st.number_input("Enter CGPA")
        #internship = st.number_input("Enter Internship")
        #projects = st.number_input("Enter Projects")
        #aptitude_score = st.number_input("Enter Aptitude Score")
        #softskills_score = st.number_input("Enter Soft Skills Score")
        #workshops = st.number_input("Enter Workshops")
        #ssc_marks = st.number_input("Enter SSC Marks")
        #hsc_marks = st.number_input("Enter HSC Marks")
                #st.write("NO : 0 and YES : 1")
        #extracurricular_activities = st.number_input("Enter Extracurricular Activities [ NO : 0  &  YES : 1 ]")
                #st.write("NO : 0 and YES : 1")
        #placement_training = st.number_input("Enter Placement Training [ NO : 0  &  YES : 1 ]")
        #if st.button("PREDICT"):
            #user_input = np.array([[cgpa,internship,projects,aptitude_score,softskills_score,workshops,ssc_marks,
                                            #hsc_marks,extracurricular_activities,placement_training]])
            #prediction = placement_data_model.predict(user_input)
            #if prediction[0] == 0:
                #st.write("Predicted Class: Not Placed")
            #else:
                #st.write("Predicted Class: Placed")


      


