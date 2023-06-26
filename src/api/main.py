from fastapi import FastAPI
import uvicorn
from utils import load_file, make_predcition
import os






# Create an instance of FastAPI
app = FastAPI(debug=True)

# get absolute path
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# set path for ml files
ml_contents_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'toolkit_folder')


ml_contents = load_file(ml_contents_path)

Encoder = ml_contents["OneHotEncoder"]
model = ml_contents["model"]





@app.get('/')
def root():
    return 'Welcome to the Gorecery Sales Forecasting API'

@app.get('/health')
def check_health():
    return {'status': 'ok'}

@app.post('/predict')
async def predict_sales(store_id: int, category_id: int, onpromotion: int, year: int,
                  month: int, dayofmonth: int, dayofweek: int, dayofyear: int,
                  weekofyear: int, quarter: int, is_month_start: int, is_quarter_start: int,
                  is_quarter_end: int, is_year_start: int, is_year_end: int, year_weekofyear: int,
                  city: str, store_type: int, cluster: int):

    input = {
    'store_id':[store_id], 
    'category_id':[category_id], 
    'onpromotion' :[onpromotion],
    'type' : [store_type], 
    'cluster': [cluster],
    'year' : [year], 
    'month' :[month], 
    'dayofmonth' :[dayofmonth],
    'dayofweek' : [dayofweek],
    'dayofyear' : [dayofyear], 
    'weekofyear' : weekofyear, 
    'quarter' : [quarter], 
    'is_month_start' : [is_month_start],
    'is_month_end' : [is_month_start], 
    'is_quarter_start' : [is_quarter_start], 
    'is_quarter_end' : [is_quarter_end], 
    'is_year_start' : [is_year_start],
    'is_year_end' : [is_year_end], 
    'year_weekofyear' : [year_weekofyear],
    'city' : [city]
    }   

 
    sales = make_predcition(Encoder, model, input)
    print(sales)
    print(type(sales[0]))
    sales_value = float(sales[0])
    return {'sales': sales_value}



if __name__ == "__main__":
    uvicorn.run('main:app', reload=True)
