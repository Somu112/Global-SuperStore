import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# importing required libraries streamlit version 0.79
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
data=pd.read_csv('Global_Superstore2.csv')
#-------------Misc
def day_segment(x):
    for  i in x:
        yield str(i[0])+'_'+str(i[1])
def color_segment(x):
    for i in x:
        if i[1]=='Consumer':
            yield '#09CDEF'
        elif i[1]=='Corporate':
            yield '#AB09EF'
        else:
            yield '#ABCD09'
            
def update_legend(fig,names):
    fig.data[0].name=names[0]
    fig.data[1].name=names[1]
    fig.data[2].name=names[2]
    return(fig)

#Initializae the evaluation dictionary
def initialize_evaluator():
    return {'Model':[],'R2':[],'MAE':[],'MSE':[],'RMSE':[]}

#Insert data in evaluation dictionary
def insert_data(test,pred,model):
    eval_data=initialize_evaluator()
    eval_data['Model'].append(model)
    eval_data['R2'].append(r2_score(test,pred))
    eval_data['MAE'].append(mean_absolute_error(test,pred))
    eval_data['MSE'].append(mean_squared_error(test,pred))
    eval_data['RMSE'].append(np.sqrt(np.absolute(mean_squared_error(test,pred))))
    return eval_data

# Append data of one dictionary to another
def append_data(data1, data2):
    for i in data1.keys():
        data2[i].extend(data1[i])
    return data2
#------------------------------------Data Processing
mean_filled_data=data.copy()
mean_filled_data['Shipping Cost'].fillna(mean_filled_data['Shipping Cost'].mean(),inplace=True)
mean_filled_data['Profit'].fillna(mean_filled_data['Profit'].mean(),inplace=True)
mean_filled_data['Discount'].fillna(mean_filled_data['Discount'].mean(),inplace=True)
# Filling with mode as Order Priority is categorical
mean_filled_data['Order Priority'].fillna(mean_filled_data['Order Priority'].mode(),inplace=True)
zero_filled_data=data.copy()
zero_filled_data['Shipping Cost'].fillna(0,inplace=True)
zero_filled_data['Profit'].fillna(0,inplace=True)
zero_filled_data['Discount'].fillna(0,inplace=True)
# Filling with mode as Order Priority is categorical
zero_filled_data['Order Priority'].fillna(zero_filled_data['Order Priority'].mode(),inplace=True)

#--- perform_eda
def perform_eda(data):
    st.title('Exploratory Data Analysis')
    st.markdown('### Univariate Distributions')
    shipcst_market=pd.DataFrame(data.groupby('Market').mean())
    fig=px.bar(shipcst_market,y=['Sales','Quantity','Discount','Profit','Shipping Cost'])


    markets=shipcst_market.index
    fig=go.Figure(data=[
        go.Bar(name='Sales', x=markets, y=shipcst_market['Sales']),
        go.Bar(name='Quantity', x=markets, y=shipcst_market['Quantity']),
        go.Bar(name='Discount', x=markets, y=shipcst_market['Discount']),
        go.Bar(name='Profit', x=markets, y=shipcst_market['Profit']),
        go.Bar(name='Shipping Cost', x=markets, y=shipcst_market['Shipping Cost'])  
    ])
    # Change the bar mode
    f=fig.update_layout(barmode='group')
    st.plotly_chart(f)
    st.markdown('''##### Observations:
- APAC has highest sales while canada makes higest profit
- EMEA has highest discount but the sales are lowest 
- Shipping cost in APAC markest is highest while in other markets its lower''')
    st.markdown('### Country Wise Sales')
    country_sales=pd.DataFrame(data.groupby('Country').mean())
    country_sales.sort_values(by='Sales',inplace=True)
    fig=px.bar(y=country_sales.Sales,x=country_sales.index,color=country_sales.Sales,
            color_continuous_scale=px.colors.sequential.Rainbow,
            height=600,width=1000)
    st.plotly_chart(fig)
    st.markdown('''##### Observations:
- Lesotho made highest sale while uganda is at lowest''')
    st.markdown('### Country Wise Profit')
    country_profit=pd.DataFrame(data.groupby('Country').mean())
    country_profit.sort_values(by='Profit',inplace=True)
    fig=px.bar(y=country_profit.Profit,x=country_profit.index,color=country_profit.Profit,
                color_continuous_scale=px.colors.sequential.Rainbow,
            height=600,width=1000)
    st.plotly_chart(fig)
    st.markdown("""#####  Obeservations
- In Lithuania the sore suffered heaviest loss while in Montenegro store made really good profit
- In 29 countries the store suffered loss""")
    st.markdown('### Sales and Profit Country Wise')
    st.plotly_chart(px.scatter(country_sales, x="Sales", y="Profit",size="Shipping Cost", color="Discount",
                 hover_name=country_sales.index, log_x=True, size_max=60))
    st.markdown('''##### Observations
- If the discount is high there will be loss
- For higher sales the shipping cost is also high''')
    st.markdown('### Monthly Sales')
    monthly_sales=data.copy()
    monthly_sales=monthly_sales.groupby('Order Date').agg(sum)
    monthly_sales.index=pd.to_datetime(monthly_sales.index)
    monthly_sales=monthly_sales.resample('M').agg(sum)
    
    fig = px.line(y=monthly_sales.Sales, x=monthly_sales.index)
    st.plotly_chart(fig)
    st.markdown('''##### Observations
- Every june, september, november and december the sales increase really high
- Every july the sales are least in the respective year''')
    st.markdown('### Yearly Analysis')
    yearly_sales=data.copy()
    yearly_sales=yearly_sales.groupby('Order Date').agg(sum)
    yearly_sales.index=pd.to_datetime(yearly_sales.index)
    yearly_sales=yearly_sales.resample('Y')
    yearly_sales=yearly_sales.agg(sum)
    fig=px.bar(yearly_sales,y=['Sales','Quantity','Discount','Profit','Shipping Cost'])


    year=yearly_sales.index
    fig=go.Figure(data=[
        go.Bar(name='Sales', x=year, y=yearly_sales['Sales']),
        go.Bar(name='Quantity', x=year, y=yearly_sales['Quantity']),
        go.Bar(name='Discount', x=year, y=yearly_sales['Discount']),
        go.Bar(name='Profit', x=year, y=yearly_sales['Profit']),
        go.Bar(name='Shipping Cost', x=year, y=yearly_sales['Shipping Cost'])  
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    st.markdown('''##### Obeservations
- The sales are increasing on yearly basis''')
    st.markdown('### Weekly Analysis')
    weekday=data.copy()
    data['Order Date']=pd.to_datetime(data['Order Date'])
    weekday['Day']=data['Order Date'].dt.day_name() 
    weekday=weekday.groupby('Day').agg(sum)
    weekday.sort_values(by='Sales',inplace=True)
    fig=px.scatter(weekday, x="Profit", y="Sales",size="Shipping Cost", color="Discount",
            hover_name=weekday.index, log_x=True, size_max=60, color_continuous_scale=px.colors.cmocean.balance)
    st.plotly_chart(fig)
    st.markdown('''##### Obsevations
- Except weekends the sales and profit made is high
- Least profit and sales made is on sunday
- Highest profit and sales made is on Friday''')
    weekday=data.copy()
    weekday['Day']=data['Order Date'].dt.day_name() 
    weekday=weekday.groupby(['Day','Segment']).agg(sum)
    weekday.sort_values(by='Sales',inplace=True)
    fig=px.scatter(weekday, x="Profit", y="Sales",size="Shipping Cost",color=list(color_segment(weekday.index)),
                hover_name=list(day_segment(weekday.index)),log_x=True, size_max=60)
    fig=update_legend(fig,['Home Office','Corporate','Consumer'])
    st.plotly_chart(fig)
    st.markdown('''##### Observations
- Consumer segment purchases more than corporate and home office and is really profitable
- Home Office segment is least profitable ''')
    categ_sales=data.groupby('Sub-Category').agg(sum)
    st.markdown('Category Sales')
    st.plotly_chart(px.scatter(categ_sales, x="Sales", y="Profit",size="Shipping Cost", color="Discount",
                 hover_name=categ_sales.index, log_x=True, size_max=60))
    st.markdown('''##### Observations
- Copier are 2nd highest selling Sub-Category but makes most of the profit
- Tables generate loss in general
''')
    st.markdown('### Country Profit vs Quantity')
    st.plotly_chart( px.scatter_geo(country_profit,locations=country_profit.index,
                    hover_name=country_profit.index,locationmode='country names',
                size=country_profit.Quantity, color=country_profit.Quantity,projection='orthographic'))
    st.markdown('''##### Observations
- Highest average quantity bought is from slovenia''')
    st.markdown('### Global Sales')
    fig = px.choropleth(country_profit, color="Sales",locationmode='country names',
                    locations=country_profit.index,)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)
    st.markdown('''##### Observations
- Except Chad, most of the countries make sales less than 400K''')

    
#----------------------------------------------------------------------------------------------Streamlit
st.title('Shipping Analysis')
perform_eda(data)