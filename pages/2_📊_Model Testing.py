import streamlit as st


if 'model' not in st.session_state:
    st.subheader("Please Train model first before continuing....")
else:

    email_classifier = st.session_state['model']
    print(email_classifier)
    # module_name = "1_Exploring_data"  # Replace with the actual module name without the ".py" extension
    # module = importlib.import_module(module_name)

    # # Now you can use the objects from the module
    # email_classifier,st = module.return_model()

    st.header("Enter email text below : ")

    msg = st.text_area('Type or paste email...',height=200)

    if msg:
        prediction = email_classifier.test_single_mail(str(msg))
        print(prediction)

        st.write("The email is predicted as : ")

        if prediction==1:
            st.error("Spam")
        else:
            st.success("Non-Spam")

        display = st.subheader("Was this email Correctly classfied?")
        container1 = st.empty()
        container2 = st.empty()
        # Add two checkboxes to the container
        yes = container1.checkbox("Yes")
        no = container2.checkbox("No")
        clicked = False
        if yes:
            st.success("Thanks for your feedback.")
            clicked = True
        if no:
            result = "Spam" if prediction == 0 else "Non-Spam"
            prediction = 0 if 1 else 1
            st.success(f"Thanks for your feedback.This is now marked as {result}")
            clicked = True
        if clicked:
            container1.empty()
            container2.empty()

            with st.spinner("Updating and Saving Model..."):
                email_classifier.update_model(prediction,msg)
            
            st.info("Model has been saved with new data points")

            
        


