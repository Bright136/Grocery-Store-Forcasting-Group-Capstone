import streamlit as st

st.set_page_config(page_icon='📉', page_title='Grocery Forcasting Api')




container = st.container()




with container:

    st.title('The Grocery Store Sales Forecasting')
    
    col1, col2, col3 = st.columns(3)
    colA, colB = st.columns(2)
    # first row
    date = col1.date_input(label='Date')
    store_id = col2.number_input(label=' Store ID')
    category_id = col3.number_input(label='Category ID')

    # second row
    store_type = col1.number_input(label='Type of Store')
    city = col2.text_input(label='City')
    holiday_type = col3.text_input(label='Type of Holiday')

    # third row
    onpromo = colA.slider(label='Number of Item On Promotion')
    cluster = colB.slider(label=' Cluster')


    button = st.button(label='Predict', use_container_width=True, type='primary')