import streamlit as st
from PIL import Image, ImageOps

################
# Introduction #
################
st.title('About Me')
st.markdown(
    """
    I have a B.S. in mathematics and a M.S. in Statistics, during my studies I worked with the USGS on predictigin the [expansion patterns of the Yellowstone grizzly bear population](https://math.montana.edu/grad_students/writing-projects/2018/JacobDym.pdf).
    My academic focus was in Bayesian statistics and the mathematics of machine learning. I've worked as a product data scientist with Intuit and most recently as the technical lead for data science on the ML platform team at Vericast.

    Most of the work I do is in a business setting and not publicly shareable, but I occasionally get time between life and work to do projects like this website.
    """
)
st.divider()

st.subheader('Skills')
st.markdown(
    """
    This website was built with Streamlit and is hosted on Google Cloud, however most of my cloud experience is on AWS.

    **Cloud:** AWS, GCP
    
    **Languages:** Python, R, SQL, HTML, Javascript, SAS, Matlab

    - **Python Libraries:** Pandas, Numpy, Scipy, Scikit-Learn, and many more!

    **Data:** Postgres, Redshift, NoSQL, MySQL, Hadoop, Hive, Spark

    **Tools & Frameworks:** Tensorflow, PyTorch, Docker, Streamlit, Jenkins, Dash/Plotly,  Tableau, QuickSight, Matillion, Djanbo, React, Terraform
    """
)

st.subheader('Hobbies')
st.markdown(
    """
    Most of my time away from work is spent with my family. The Wood River Valley has great outdoor access throughout the year. I hike, fly fish, ski, snowboard, and nordic ski while bring my dog on many a journey along the way.
    """
)
col1, col2 = st.columns(2)
with col1:
    tate_image = Image.open('media/tate_image.JPG')
    st.image(tate_image, width=500)
with col2:
    ski_image = Image.open('media/touring_image.jpg')
    st.image(ski_image, width=500)

st.subheader('Volunteering')
st.markdown(
    """
    Most of my volunteer time goes to being a volunteer firefighter. My small town still relies primarily on volunteers to provide emergency response services, and so that's where the remainder of my time goes.
    """
)