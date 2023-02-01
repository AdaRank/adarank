import streamlit as st

st.title("Help Page")
st.markdown("### File Format Example:")
st.markdown("In the following we give an example of a file containing items:")
st.markdown('''1 -1 2 -1 5 -1 3 -1 -2''')
st.markdown('''4 -1 3 -1 1 -1 -2''')
st.markdown('''5 -1 6 -1 3 -1 1 -1 7 -1 -2''')
st.markdown("---")
st.markdown("If CGap is applied another file with a timestamp matching the position of the corresponding item is required, e.g. (time in ms):")
st.markdown('''966020147 -1 966020182 -1 966020182 -1 966020182 -1 -2''')
st.markdown('''1033238124 -1 1043901943 -1 1043902000 -1 -2''')
st.markdown('''1011752197 -1 1011752218 -1 1015042992 -1 1015211668 -1 1034768334 -1 -2''')

