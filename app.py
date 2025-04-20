import pickle
from PIL import Image

LogReg_model=pickle.load(open('LogReg.pkl','rb'))
DecisionTree_model=pickle.load(open('DecisionTree.pkl','rb'))
NaiveBayes_model=pickle.load(open('NaiveBayes.pkl','rb'))
RF_model=pickle.load(open('RF.pkl','rb'))

def classify(answer):
    return answer[0]+" is the best crop for cultivation in this condition."


def main():
    st.title("(Crop Recommendation using ML and IOT)...")
    html_temp = """
    <div style="background-color:teal; padding:20px">
    <h2 style="color:white;text-align:center;">Best crop for cultivation</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Naive Bayes (Accuracy: 98.86%)','Logistic Regression (Accuracy: 90.68%)','Decision Tree (Accuracy: 90.68%)','Random Forest (Accuracy: 99.54%)']
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
        if option=='Logistic Regression':
            st.success(classify(LogReg_model.predict(inputs)))
        elif option=='Decision Tree':
            st.success(classify(DecisionTree_model.predict(inputs)))
        elif option=='Naive Bayes':
            st.success(classify(NaiveBayes_model.predict(inputs)))
        else:
            st.success(classify(RF_model.predict(inputs)))   


if __name__=='__main__':
    main()
