import streamlit as st
import pickle
from PIL import Image

svm_model=pickle.load(open('SVM.pkl','rb'))
DecisionTree_model=pickle.load(open('DecisionTree.pkl','rb'))
NaiveBayes_model=pickle.load(open('NaiveBayes.pkl','rb'))
RF_model=pickle.load(open('RF.pkl','rb'))

def classify(answer):
    return answer[0]+" is the best crop for cultivation in this condition."


def main():
    st.title("Crop Recommendation using ML and IOT")
    

    activities=['Naive Bayes (Accuracy: 98.86%)','Decision Tree (Accuracy: 90.68%)','SVM (Accuracy: 97.72%)','Random Forest (Accuracy: 99.54%)']
    option=st.sidebar.selectbox("Choose model?",activities)
    st.subheader(option)
    sn=st.slider('NITROGEN (N)', 0.0, 200.0)
    sp=st.slider('PHOSPHOROUS (P)', 0.0, 200.0)
    pk=st.slider('POTASSIUM (K)', 0.0, 200.0)
    pt=st.slider('TEMPERATURE', 0.0, 50.0)
    phu=st.slider('HUMIDITY', 0.0, 100.0)
    pPh=st.slider('Ph', 0.0, 14.0)
    pr=st.slider('RAINFALL', 0.0, 300.0)
    inputs=[[sn,sp,pk,pt,phu,pPh,pr]]
    if st.button('Predict'):
        if option=='SVM (Accuracy: 97.72%)':
            st.success(classify(svm_model.predict(inputs)))
        elif option=='Decision Tree (Accuracy: 90.68%)':
            st.success(classify(DecisionTree_model.predict(inputs)))
        elif option=='Naive Bayes (Accuracy: 98.86%)':
            st.success(classify(NaiveBayes_model.predict(inputs)))
        else:
            st.success(classify(RF_model.predict(inputs)))   


if __name__=='__main__':
    main()
