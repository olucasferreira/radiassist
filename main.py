import query_data as qd
import streamlit as st

st.title("RadiAssist")
st.subheader("Orientação inteligente para solicitação de exames de imagem conforme diretrizes da ACR")

with st.form("my_form"):

    raw_text = st.text_area("Descreva as características do paciente, como idade, sexo, sintomas, histórico médico relevante, e quaisquer outros detalhes relevantes para a avaliação do pedido do exame:", max_chars=1000)
    submitted = st.form_submit_button("Enviar")
    
    if submitted:
        response, context = qd.main(query_text=raw_text)
        st.write(response)
        expander = st.expander("Referências")
        expander.write(context)