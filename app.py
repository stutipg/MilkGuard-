import streamlit as st
import pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="MilkGuard",page_icon="🥛",layout="wide")
BASE=Path(__file__).parent
model=joblib.load(BASE/"milkguard_model.pkl")
df=pd.read_csv(BASE/"data/MilkGuard_simulated_features.csv")
FEATURES=["R","G","B","H","S","V","Brightness","Contrast"]

st.title("🥛 MilkGuard")
st.subheader("AI-powered preliminary milk adulteration screening — digital MVP")
st.warning("DEMO / SIMULATION: this version uses synthetic feature data. It is NOT validated for real milk testing and is not a laboratory replacement.")

tab1,tab2,tab3=st.tabs(["🔬 Analyze","📊 Model Evidence","ℹ️ How it works"])
with tab1:
    st.markdown("### Simulated sample analysis")
    choice=st.selectbox("Choose a prepared demo sample",["Pure milk","Water-diluted milk","Starch-adulterated milk"])
    mp={"Pure milk":"Pure","Water-diluted milk":"Water Suspected","Starch-adulterated milk":"Starch Suspected"}
    s=df[df.Class==mp[choice]].sample(1).iloc[0]
    c1,c2=st.columns(2)
    with c1:
        st.info("Simulated image-derived features")
        st.dataframe(pd.DataFrame([s[FEATURES].astype(float).round(2).to_dict()]),hide_index=True,use_container_width=True)
    with c2:
        x=pd.DataFrame([[s[f] for f in FEATURES]],columns=FEATURES)
        p=model.predict_proba(x)[0]; pred=model.classes_[int(np.argmax(p))]
        st.success(f"Result: **{pred}**")
        st.metric("Demo model confidence",f"{max(p)*100:.1f}%")
        st.caption("This confidence is from the synthetic proof-of-concept model.")
    st.markdown("**Recommendation:** flagged samples require reference/laboratory verification.")

with tab2:
    st.markdown("### Synthetic hold-out evaluation")
    X=df[FEATURES]; y=df.Class
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    pv=model.predict(Xv)
    st.metric("Hold-out accuracy (synthetic data)",f"{accuracy_score(yv,pv)*100:.1f}%")
    st.image(str(BASE/"assets/confusion_matrix.png"),caption="Synthetic-data evaluation")
    st.caption("Replace this with real, independently collected milk-image results before claiming performance.")

with tab3:
    st.markdown("""### Proposed real-world pipeline
**Milk sample → controlled LED enclosure → smartphone image → OpenCV → RGB/HSV/brightness/texture features → ML classifier → screening result**

### Current MVP
This online prototype demonstrates the software/AI workflow using synthetic numerical features representing image-derived measurements.

### Next validation stage
Collect multiple real milk batches, prepare labelled samples, photograph them under fixed illumination, extract real image features, train with batch-separated validation, and compare flagged samples against reference tests.

**Positioning:** rapid first-level screening aid, not a definitive laboratory test.
""")
