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
    import streamlit as st

    # Create a button to clear the page
    sub_button = st.button("Submit")
    if sub_button not in st.session_state:
        st.session_state['sub_button'] = True
    # Check if the button is clicked

    if sub_button or msg:
        
        prediction = email_classifier.test_single_mail(str(msg))
        print(prediction)

        placeholder = st.empty()
        with placeholder.container():
            st.write("The email is predicted as : ")

            if prediction==1:
                st.error("Spam")
            else:
                st.success("Non-Spam")


            st.subheader("Was this email Correctly classfied?")
            
            box1 = st.empty()
            box2 = st.empty()
            # Add two checkboxes to the container
            yes = box1.button("  Yes   ")
            no = box2.button("   No   ")
            clicked = False
            if yes:
                st.success("Thanks for your feedback.")
                clicked = True
            if no:
                result = "Spam" if prediction == 0 else "Non-Spam"
                prediction = 0 if prediction == 1 else 1
                print("-----------Prediction=",prediction)
                st.success(f"Thanks for your feedback.This is now marked as {result}")
                clicked = True
            if clicked:
                print("---------------alrady clicked here--------------")

                box1.empty()
                box2.empty()
                clicked = False

                with st.spinner("Updating and Saving Model..."):
                    email_classifier.update_model(prediction,msg)
                
                info_msg = st.info("Model has been saved with new data points")
        
                

            
        


