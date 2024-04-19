# Snow Depth Prediction
The aim of this project is to predict snow depth in Finnish ski centers by using historical data to train machine learning models.

## Project components
1. Data collection: Data collected from FMI* and NASA's POWER project**
2. Data pipeline: Clean and modify data using Python
3. Machine Learning: Models trained using scikit-learn, future data extrapolated using Prophet
4. Data visualisation: Figures created using Matplotlib

*temperature, cloud cover, snow depth<br>
**solar radiation

![figure1: project steps](https://github.com/vltnnx/Snow-Depth-Prediction/blob/main/fig/project_steps.png?raw=true)

## Visualisations - Tableau
![figure2: historical and prediction data visualised](https://raw.githubusercontent.com/vltnnx/Snow-Depth-Prediction/main/fig/fig-tableau.png)


<div class='tableauPlaceholder' id='viz1713534925426' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sn&#47;SnowDepthPredictionProject&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='SnowDepthPredictionProject&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sn&#47;SnowDepthPredictionProject&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1713534925426');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1327px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
