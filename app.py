import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'best_model2.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ACNE_CLASSES = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

ACNE_INFO = {
    'Blackheads': {
        'precautions': ["Cleanse Twice Daily - Wash your face in the morning and night with a mild cleanser.",
                        "Avoid Heavy Makeup - Use oil-free and non-comedogenic products.",
                        "Change Pillowcases Often - Bacteria and oils on pillowcases can clog pores.",
                        "Don't Squeeze Blackheads - This can cause scarring and infection.",
                        "Use Sunscreen - Protect your skin with an oil-free, non-comedogenic sunscreen.",
                        "Eat a Healthy Diet - Reduce sugar and dairy, as they may trigger excess oil production."
                        ],                
    },
    'Cyst': {
        'precautions': ["Avoid picking or popping", 
                        "Keep skin clean", 
                        "Use mild, fragrance-free products",
                        "Apply ice to reduce swelling",
                        "Avoid sun exposure",
                        "Consult a dermatologist"
                        ]
    },
    'Papules': {
        'precautions': ["Use gentle cleansers",
                        "Avoid harsh scrubbing", 
                        "Apply tea tree oil",
                        "Use oil-free moisturizers",
                        "Avoid touching your face",
                        "Consult a dermatologist"
                        ],
    },
    'Pustules': {
        'precautions': ["Avoid touching affected areas", 
                        "Use lightweight, oil-free products", 
                        "Change pillowcases regularly",
                        "Wash makeup brushes weekly",
                        "Use sunscreen",
                        "Consult a dermatologist"
                        ]
    },
    'Whiteheads': {
        'precautions': ["Exfoliate regularly with AHA/BHA", 
                        "Use non-comedogenic moisturizer", 
                        "Wash makeup brushes weekly",
                        "Use sunscreen",
                        "Consult a dermatologist",
                        "Avoid picking or squeezing",
                        "Keep skin clean"
                        ]
    }
}

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

PRODUCTS = {
    'Blackheads': ('images/blackheads.png', 'https://www.amazon.in/XYST-Certified-Tightening-Detoxifies-Hyaluronic/dp/B0B34WDWST/ref=sr_1_2_sspa?crid=3MVRBHKKW6KOE&dib=eyJ2IjoiMSJ9.IJdYrOvBA5kgXrgKlPY4phnRJmkLl9slpvm2mGFRj0HQxTxLRo5S8y1nGDAtbGKH-zx9Nyk2ZkPP7ppeDPk9cQ8gipK0ZaHrQ_SZpewL277ioU2_WwWYTyexNtvgGIVB4cegGJtf9YPtGCF4eZeiSMPNzAzXCP1timBA_PwHwO-rLrlJeq_-aKbHggtRHNwXS1mBT7PDupWw-mdjEKgcIYco3_AXsqc2swKBIYu8SHhfjyEfGC-Omk1xoQiiJiks-pG_qU7jLUdIWabX3_oVDprr81Cw3B7s1b8SQs8C1uU.h_E4LKzAvYCh3JI_UHHtR78E8RffWZH442IbwspSPvk&dib_tag=se&keywords=blackheads+removal+mask&qid=1740202991&sprefix=blackheads+%2Caps%2C238&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'),
    
    'Cyst': ('images/cyst.jpg', 'https://www.amazon.in/Rawls-Acne-Corrector-Breakout-goodness/dp/B0CXPRJKT2/ref=sr_1_2_sspa?crid=VXEUMYVF1VA4&dib=eyJ2IjoiMSJ9.AV0bE6W-74aVwSmkMqeDy0ZNFncfVfREscC2tyn2gI_87IYcDZSwshK0GAj_4PE_qBrLeG6LC0WEY0_etLYgk943hLyEQa9rcEvjenhYX3ndRlKP2uxPPNZmT0cWWfg1kO6Nu3wCpDhNqHBnhf_5HZSDfVcvF76VTEe9Db0MQlyFOzDBgFZA-mNf9SkxeJJig3hY8_t53qFqJ8jcX_8ijhM_JNGYJGKChrHBSBzmwp4QckEvAW89jalgs-RaniPRi33oYcXWQrmUUNkBBfNAbYwrJt6YmeCXD86xCy20sL4._KsT_bbfHo0VOO2xdcWEWDVHeHHE_Z1bbc4hcOQTwDw&dib_tag=se&keywords=benzoyl+peroxide+cream&qid=1740203380&sprefix=benzoyl+peroxide+crea%2Caps%2C256&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'),
    
    'Papules': ('images/papules.jpg', 'https://www.amazon.in/Fixderma-Salyzap-Gel-Time-20ml/dp/B089RJXHP5/ref=sr_1_7?crid=2SX75VM7MI4ZO&dib=eyJ2IjoiMSJ9.lQbhdUXTKvHRx_5kjxZYyGS34hKJlhg2y40z9po72DibYmqMRCVH_91V8nEY-lr6Iu_S84mq9hC8C32kDkCyDmDttdyPRwyuOSfdVSsMNw0SGgpxkYiXCJthx_oNGOE-jrPqGPu6UyWQL4CN3Z3q3uDoFylr7ZyPB97ksC9qB_Kur8-ijs5vhYvsATxdyWNqYrPlquoUmpRqt1smIFv6i-ZqJXqNotLC6t_xxcVFEPwk4o0pnVwBkVSHVtKIHsJ1Rqb-8LLEuaOHL0WosD9HK5evapHKqCcYPR2J0K4jkbmJrd5N2WloGJ9MRm__tTJhdvGm9QlPR97MZtigBApKX46sLhfKCbriE9cCXTAx0BeazrZSdyCoWWYcCkrzRrt0O30ChAr8knIcSCVAa6SOyNfRiiWRzodkJrBN9DVzoARCbgfGib8YWPz8Wed5rSKe.VepDmYM8zoQpHuoO1bcw2fGk8jIipeg9kIY4mzwtwek&dib_tag=se&keywords=clindamycin+and+benzoyl+peroxide+gel&qid=1740207170&sprefix=Clindamycin%2Caps%2C284&sr=8-7'),
    
    'Pustules': ('images/pustules.jpg', 'https://www.amazon.in/Eczemaron-Ayurvedic-ItchCoat-Ringworm-itching/dp/B0DQCKKF8N/ref=sr_1_15?crid=2S6WHKMH2DTLI&dib=eyJ2IjoiMSJ9.EMUQU4clhkBI2x_80MI_Yk5LgbCIjX51TivPCkFeZPIFFQ6VRyjE27gvRHpKWt9QTOgWzJ-NgBsB1UbxcjjMUzQi_O8J7gd5LX-afAefwc8tg8mRKQY8nITLk-MnQnC8gpt0EyfhwtLue0we7YwVE482KH08eu3CiU6svCuCNe_EBM67MzKGmGralwhPT-xn0tetJ6i1SV6oiPOyHwA5IT_TdRTjhjPSb34GYcmSuFWs3WEuscFvHT0BMnc0ysFilheTt4o2Ic7gou09nz3Hc7ARZmkST_NAqYRGwg88HjYaD8n_iO7HQqrHuKHOW7O1pF8OX2DljOa7aZhg8zgLPASUs1Au76KRsPxFwp6m4J7NOL8RBjMbQsO_nsxmsBS8rqzhnGEL26t688gX5g2irMijLsa5eChFH7beAkdDBwE0ytyNV1kdLL_4QpVppSWS.N60sqyWHPTRBFCP3EkBDNStynI4xfGuGasxuD68B1gw&dib_tag=se&keywords=antifungal+cream&qid=1740207430&sprefix=antif%2Caps%2C282&sr=8-15'),
    
    'Whiteheads': ('images/whiteheads.jpg', 'https://www.amazon.in/Minimalist-Wrinkles-Bakuchiol-Anti-Aging-Hydrating/dp/B0DK954QXH/ref=sr_1_2_sspa?crid=3EAC9JFD6GR3O&dib=eyJ2IjoiMSJ9.j5V7WfLrZWqJdWvaV4xdsHXjJfzd7HHjFJoibS8e-es615dBXOPZ3E0k9Er7SFPWNbmH_IgT8BzFfSb06hr-GFyY3N5e9zeKQ31-zy4kVmnFnyam2frRkdg3flycDW0ORatxX9KMg6zQBeucQ3ox4_-2FavfqZeC5gh5tqA4BtPG-ugt_DpGMf_Z6HsVCQNGIbDqq7BkuS39Sse5OQVq2kayCAbtw57jIBYD11FhnQYuvDa1xQizP2BXRPEGaxhnkaI2nFMzkgE32nWxUfVGPdqDBCX2SOz9jl8mlt5YUQs.9LZxf-YGn76nGy12wF53vpWPAl4WUdW1eyLz9oMAS7w&dib_tag=se&keywords=retinol+serum+for+whiteheads&qid=1740203671&sprefix=retinol+serum+for+whitehead%2Caps%2C247&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'),
}

def predict_image(image_array):
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  

def plot_acne_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ACNE_CLASSES, probabilities * 100, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF6666'])
    
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Acne Type Prediction Confidence")
    ax.invert_yaxis()  

    for i, v in enumerate(probabilities * 100):
        ax.text(v + 1, i, f"{v:.2f}%", va='center')

    st.pyplot(fig)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Acne Detection", "Products"])

if page == "Acne Detection":
    st.title("Acne Detection System")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

        st.write("Processing...")
        processed_image = preprocess_image(image)
        image.resize((300, 300))
        probabilities = predict_image(processed_image)

        plot_acne_probabilities(probabilities)

        top_prediction_idx = np.argmax(probabilities)
        top_prediction = ACNE_CLASSES[top_prediction_idx]
        confidence = float(probabilities[top_prediction_idx])
        st.write(f"**Top Prediction: {top_prediction} ({confidence * 100:.2f}% confidence)**")

        st.subheader("Precautions")
        for precaution in ACNE_INFO[top_prediction]['precautions']:
            st.write(f"- {precaution}")

        product_image, buy_link = PRODUCTS[top_prediction]
        st.subheader("**Recommended Product**")
        product_image = Image.open(product_image)
        product_image = product_image.resize((300, 300))
        st.image(product_image)
        st.markdown(f"[Buy Now]({buy_link})", unsafe_allow_html=False)

elif page == "Products":
    st.title("Recommended Skincare Products")

    products = [
        {
            "category": "For Blackheads",
            "image": "images/blackheads.png",
            "name": "XYST Activated Charcoal Clay Mask Face Mask",
            "link": "https://www.amazon.in/XYST-Certified-Tightening-Detoxifies-Hyaluronic/dp/B0B34WDWST/ref=sr_1_2_sspa?crid=3MVRBHKKW6KOE&dib=eyJ2IjoiMSJ9.IJdYrOvBA5kgXrgKlPY4phnRJmkLl9slpvm2mGFRj0HQxTxLRo5S8y1nGDAtbGKH-zx9Nyk2ZkPP7ppeDPk9cQ8gipK0ZaHrQ_SZpewL277ioU2_WwWYTyexNtvgGIVB4cegGJtf9YPtGCF4eZeiSMPNzAzXCP1timBA_PwHwO-rLrlJeq_-aKbHggtRHNwXS1mBT7PDupWw-mdjEKgcIYco3_AXsqc2swKBIYu8SHhfjyEfGC-Omk1xoQiiJiks-pG_qU7jLUdIWabX3_oVDprr81Cw3B7s1b8SQs8C1uU.h_E4LKzAvYCh3JI_UHHtR78E8RffWZH442IbwspSPvk&dib_tag=se&keywords=blackheads+removal+mask&qid=1740202991&sprefix=blackheads+%2Caps%2C238&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"
        },
        {
            "category": "For Cystic Acne",
            "image": "images/cyst.jpg",
            "name": "Benzoyl Peroxide Cream",
            "link": "https://www.amazon.in/Rawls-Acne-Corrector-Breakout-goodness/dp/B0CXPRJKT2/ref=sr_1_2_sspa?crid=VXEUMYVF1VA4&dib=eyJ2IjoiMSJ9.AV0bE6W-74aVwSmkMqeDy0ZNFncfVfREscC2tyn2gI_87IYcDZSwshK0GAj_4PE_qBrLeG6LC0WEY0_etLYgk943hLyEQa9rcEvjenhYX3ndRlKP2uxPPNZmT0cWWfg1kO6Nu3wCpDhNqHBnhf_5HZSDfVcvF76VTEe9Db0MQlyFOzDBgFZA-mNf9SkxeJJig3hY8_t53qFqJ8jcX_8ijhM_JNGYJGKChrHBSBzmwp4QckEvAW89jalgs-RaniPRi33oYcXWQrmUUNkBBfNAbYwrJt6YmeCXD86xCy20sL4._KsT_bbfHo0VOO2xdcWEWDVHeHHE_Z1bbc4hcOQTwDw&dib_tag=se&keywords=benzoyl+peroxide+cream&qid=1740203380&sprefix=benzoyl+peroxide+crea%2Caps%2C256&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"
        },
        {
            "category": "For Whiteheads",
            "image": "images/whiteheads.jpg",
            "name": "Retinol Serum",
            "link": "https://www.amazon.in/Minimalist-Wrinkles-Bakuchiol-Anti-Aging-Hydrating/dp/B0DK954QXH/ref=sr_1_2_sspa?crid=3EAC9JFD6GR3O&dib=eyJ2IjoiMSJ9.j5V7WfLrZWqJdWvaV4xdsHXjJfzd7HHjFJoibS8e-es615dBXOPZ3E0k9Er7SFPWNbmH_IgT8BzFfSb06hr-GFyY3N5e9zeKQ31-zy4kVmnFnyam2frRkdg3flycDW0ORatxX9KMg6zQBeucQ3ox4_-2FavfqZeC5gh5tqA4BtPG-ugt_DpGMf_Z6HsVCQNGIbDqq7BkuS39Sse5OQVq2kayCAbtw57jIBYD11FhnQYuvDa1xQizP2BXRPEGaxhnkaI2nFMzkgE32nWxUfVGPdqDBCX2SOz9jl8mlt5YUQs.9LZxf-YGn76nGy12wF53vpWPAl4WUdW1eyLz9oMAS7w&dib_tag=se&keywords=retinol+serum+for+whiteheads&qid=1740203671&sprefix=retinol+serum+for+whitehead%2Caps%2C247&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"
        },
        {
            "category": "For Pustules",
            "image": "images/pustules.jpg",
            "name": "Antifungal Cream",
            "link": "https://www.amazon.in/Eczemaron-Ayurvedic-ItchCoat-Ringworm-itching/dp/B0DQCKKF8N/ref=sr_1_15?crid=2S6WHKMH2DTLI&dib=eyJ2IjoiMSJ9.EMUQU4clhkBI2x_80MI_Yk5LgbCIjX51TivPCkFeZPIFFQ6VRyjE27gvRHpKWt9QTOgWzJ-NgBsB1UbxcjjMUzQi_O8J7gd5LX-afAefwc8tg8mRKQY8nITLk-MnQnC8gpt0EyfhwtLue0we7YwVE482KH08eu3CiU6svCuCNe_EBM67MzKGmGralwhPT-xn0tetJ6i1SV6oiPOyHwA5IT_TdRTjhjPSb34GYcmSuFWs3WEuscFvHT0BMnc0ysFilheTt4o2Ic7gou09nz3Hc7ARZmkST_NAqYRGwg88HjYaD8n_iO7HQqrHuKHOW7O1pF8OX2DljOa7aZhg8zgLPASUs1Au76KRsPxFwp6m4J7NOL8RBjMbQsO_nsxmsBS8rqzhnGEL26t688gX5g2irMijLsa5eChFH7beAkdDBwE0ytyNV1kdLL_4QpVppSWS.N60sqyWHPTRBFCP3EkBDNStynI4xfGuGasxuD68B1gw&dib_tag=se&keywords=antifungal+cream&qid=1740207430&sprefix=antif%2Caps%2C282&sr=8-15"
        },
        {
            "category": "For Papules",
            "image": "images/papules.jpg",
            "name": "Clindamycin and Benzoyl peroxide gel",
            "link": "https://www.amazon.in/Fixderma-Salyzap-Gel-Time-20ml/dp/B089RJXHP5/ref=sr_1_7?crid=2SX75VM7MI4ZO&dib=eyJ2IjoiMSJ9.lQbhdUXTKvHRx_5kjxZYyGS34hKJlhg2y40z9po72DibYmqMRCVH_91V8nEY-lr6Iu_S84mq9hC8C32kDkCyDmDttdyPRwyuOSfdVSsMNw0SGgpxkYiXCJthx_oNGOE-jrPqGPu6UyWQL4CN3Z3q3uDoFylr7ZyPB97ksC9qB_Kur8-ijs5vhYvsATxdyWNqYrPlquoUmpRqt1smIFv6i-ZqJXqNotLC6t_xxcVFEPwk4o0pnVwBkVSHVtKIHsJ1Rqb-8LLEuaOHL0WosD9HK5evapHKqCcYPR2J0K4jkbmJrd5N2WloGJ9MRm__tTJhdvGm9QlPR97MZtigBApKX46sLhfKCbriE9cCXTAx0BeazrZSdyCoWWYcCkrzRrt0O30ChAr8knIcSCVAa6SOyNfRiiWRzodkJrBN9DVzoARCbgfGib8YWPz8Wed5rSKe.VepDmYM8zoQpHuoO1bcw2fGk8jIipeg9kIY4mzwtwek&dib_tag=se&keywords=clindamycin+and+benzoyl+peroxide+gel&qid=1740207170&sprefix=Clindamycin%2Caps%2C284&sr=8-7"
        }
    ]

    for product in products:
        with st.container():
            st.subheader(product["category"])
            image = Image.open(product["image"])
            resized_image = image.resize((300, 300))
            st.image(resized_image)
            st.write(f"**{product['name']}**")
            st.markdown(f"[Buy Now]({product['link']})", unsafe_allow_html=True)