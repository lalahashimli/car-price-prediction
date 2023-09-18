from PIL import Image
import streamlit as st
import pandas as pd
import warnings
import pickle
import numpy as np
import time
# from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold,StratifiedKFold,train_test_split,GridSearchCV,cross_val_score
# from sklearn.preprocessing import StandardScaler , RobustScaler, MaxAbsScaler,MinMaxScaler,OneHotEncoder, LabelEncoder
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer

df = pd.read_csv('turbo.csv')

df.columns = df.columns.str.replace(pat = ' ', repl = '_')
df.columns = df.columns.str.lower()
df.columns = df.columns.str.strip()

df = df.drop(columns= ['satici','telefonlar','yeniləndi','baxışların_sayı','url','sahiblər','qəzalı','etrafli','yerlərin_sayı'])

df=df.loc[df['qiymet']>1000]

def convert_currency(df = None):
    azn_index = df.loc[df.valyuta =='AZN'].index
    euro_index = df.loc[df.valyuta == 'EUR'].index
    usd_index = df.loc[df.valyuta == 'USD'].index
    
    euro_to_azn = df.loc[df.index.isin(values = euro_index), 'qiymet'].astype(dtype = 'float') * 1.81
    usd_to_azn = df.loc[df.index.isin(values = usd_index), 'qiymet'].astype(dtype = 'float') * 1.7
    azn = df.loc[df.index.isin(values = azn_index), 'qiymet'].astype(dtype = 'float')
    
    df.loc[df.index.isin(values = euro_index), 'qiymet'] = euro_to_azn
    df.loc[df.index.isin(values = usd_index), 'qiymet'] = usd_to_azn
    df.loc[df.index.isin(values = azn_index), 'qiymet'] = azn
    
    df.qiymet = df.qiymet.apply(func = lambda x: int(x))
    df.drop(columns = 'valyuta', inplace = True)
    return df

def create_new_columns(df = None):
    df['mühərrik_hecmi'] = df.mühərrik.apply(lambda x: x.split('/')[0])
    df['mühərrik_gucu'] = df.mühərrik.apply(lambda x: x.split('/')[1])
    df['yanacaq_novu'] = df.mühərrik.apply(lambda x: x.split('/')[2])
    df['vuruğu_var'] = df.vəziyyəti.apply(lambda x: x.split(',')[0] if pd.notna(x) else np.nan)
    df['rənglənib'] = df.vəziyyəti.apply(lambda x: x.split(',')[1] if pd.notna(x) else np.nan)
    df['lehimli_disk'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Yüngül lehimli disklər' in x else 'no')
    df['abs_'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'ABS' in x else 'no')
    df['lyuk'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Lyuk' in x else 'no')
    df['yağış_sensoru'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Yağış sensoru' in x else 'no')
    df['mərkəzi_qapanma'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Mərkəzi qapanma' in x else 'no')
    df['park_radarı'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Park radarı' in x else 'no')
    df['kondisioner'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Kondisioner' in x else 'no')
    df['oturacaqların_isidilməsi'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Oturacaqların isidilməsi' in x else 'no')
    df['dəri_salon'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Dəri salon' in x else 'no')
    df['ksenon_lampalar'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Ksenon lampalar' in x else 'no')
    df['arxa_görüntü_kamerası'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Arxa görüntü kamerası' in x else 'no')
    df['yan_pərdələr'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Yan pərdələr' in x else 'no')
    df['oturacaqların_ventilyasiyası'] = df.extra.apply(lambda x: np.nan if pd.isna(x) else 'yes' if 'Oturacaqların ventilyasiyası' in x else 'no')

    df.drop(columns = ['mühərrik','vəziyyəti','extra'], inplace = True)
    return df

def convert_int(df = None):
    df.mühərrik_hecmi = df.mühərrik_hecmi.str.replace(pat = 'L', repl = '')
    df.mühərrik_gucu = df.mühərrik_gucu.str.replace(pat = 'a.g.', repl = '')
    df.yürüş = df.yürüş.str.replace(pat = 'km', repl = '')
    df.yürüş = df.yürüş.str.replace(pat = ' ', repl = '')
    df.mühərrik_hecmi = pd.to_numeric(arg = df.mühərrik_hecmi, downcast = 'float')
    df[['mühərrik_gucu', 'yürüş']] = df[['mühərrik_gucu', 'yürüş']].applymap(func = lambda x: int(x))
    return df

def convert_str(df = None):
    df.avtosalon = df.avtosalon.apply(lambda x: 'he' if x==1 else 'yox')
    return df

def convert_lower_case(df = None):
    df_obj = df.select_dtypes(include = 'object') 
    df[df_obj.columns] = df_obj.applymap(lambda x: np.nan if pd.isna(x) else x.lower())
    return df

def replace_value(df=None):
    df['yeni'] =  df['yeni'].str.strip().replace({'bəli': 'yes', 'xeyr': 'no'})
    return df

df = df.pipe(func = convert_currency).pipe(func = create_new_columns).pipe(func = convert_int).pipe(func = convert_str).pipe(func = convert_lower_case).pipe(func = replace_value)


moto_nan_list=df[df['ban_növü']=='motosiklet'].iloc[:,19:31].drop(columns=['dəri_salon','abs_']).columns.tolist()


def motosikle_change_nan(data_frame=None):
    for i in moto_nan_list:
        data_frame[i].fillna('no' , inplace=True)
    return data_frame
motosikle_change_nan(data_frame=df)


categoric_data = df.select_dtypes(include='object')

imputer = SimpleImputer(strategy='most_frequent')

categoric_data_imputed = pd.DataFrame(imputer.fit_transform(categoric_data), columns=categoric_data.columns)

df[categoric_data.columns] = categoric_data_imputed

df=pd.concat([categoric_data,df.select_dtypes(include='number')] , axis=1)

    
interface = st.container()


with interface:
    
    label_encoder = LabelEncoder()

    
    marka_encoding = label_encoder.fit_transform(df['marka'])
    marka_mapping = {name: value for name, value in zip(df.marka.tolist(), marka_encoding)}

    model_encoding = label_encoder.fit_transform(df['model'])
    model_mapping = {name: value for name, value in zip(df.model.tolist(), model_encoding)}

    şəhər_encoding = label_encoder.fit_transform(df['şəhər'])
    şəhər_mapping = {name: value for name, value in zip(df.şəhər.tolist(), şəhər_encoding)}

    yanacaq_novu_encoding = label_encoder.fit_transform(df['yanacaq_novu'])
    yanacaq_novu_mapping = {name: value for name, value in zip(df.yanacaq_novu.tolist(), yanacaq_novu_encoding)}

    ötürücü_encoding = label_encoder.fit_transform(df['ötürücü'])
    ötürücü_mapping = {name: value for name, value in zip(df.ötürücü.tolist(), ötürücü_encoding)}

    ban_növü_encoding = label_encoder.fit_transform(df['ban_növü'])
    ban_növü_mapping = {name: value for name, value in zip(df.ban_növü.tolist(), ban_növü_encoding)}

    sürətlər_qutusu_encoding = label_encoder.fit_transform(df['sürətlər_qutusu'])
    sürətlər_qutusu_mapping = {name: value for name, value in zip(df.sürətlər_qutusu.tolist(), sürətlər_qutusu_encoding)}

    rəng_encoding = label_encoder.fit_transform(df['rəng'])
    rəng_mapping = {name: value for name, value in zip(df.rəng.tolist(), rəng_encoding)}

    hansı_bazar_encoding = label_encoder.fit_transform(df['hansı_bazar_üçün_yığılıb'])
    hansı_bazar_mapping = {name: value for name, value in zip(df.hansı_bazar_üçün_yığılıb.tolist(), hansı_bazar_encoding)}

   
   
    st.title(body = 'Enter Key Car Features')
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
    
   
    marka , model, şəhər = st.columns(spec = [1, 1, 1])
    

    with marka:
        marka = st.selectbox(label = 'Brand', options = df['marka'].sort_values().unique().tolist())
    
    with model:
        model = st.selectbox(label = 'Model', options = df[df['marka'] == marka]['model'].sort_values().unique().tolist())
        
    with şəhər:
        şəhər = st.selectbox(label = 'City', options = df['şəhər'].sort_values().unique().tolist())
        
    st.markdown(body = '***')
    
    
    yanacaq_novu, ötürücü, ban_növü, sürətlər_qutusu = st.columns(spec = [1, 1, 1, 1])
        
    with yanacaq_novu:
        yanacaq_novu = st.selectbox(label = 'Fuel type', options = df.yanacaq_novu.unique().tolist())
    
    with ötürücü:
        ötürücü = st.selectbox(label = 'Gear', options = df.ötürücü.unique().tolist())

    with ban_növü:
        ban_növü = st.selectbox(label = 'Ban type', options = df.ban_növü.unique().tolist())
        
    with sürətlər_qutusu:
        sürətlər_qutusu = st.selectbox(label = 'Gear box', options = df.sürətlər_qutusu.unique().tolist())
        
        
    
    yürüş = st.number_input(label = 'Mileage (km)', value = 0, step = 10 )
    button_text = 'Send values'
 
    st.markdown(body = '***')
    
    buraxılış_ili = st.slider(label='Year',min_value = int(df.buraxılış_ili.min()),
                              max_value= int(df.buraxılış_ili.max()),value = int(df.buraxılış_ili.mean()))
        
    rəng, hansı_bazar_üçün_yığılıb = st.columns(spec = [1, 1])
    
    with rəng:
         rəng = st.selectbox(label = 'Color', options = df.rəng.sort_values().unique().tolist())
            
    with hansı_bazar_üçün_yığılıb:
         hansı_bazar_üçün_yığılıb = st.selectbox(label = 'For which market it is assembled', options = categoric_data_imputed.hansı_bazar_üçün_yığılıb.sort_values().unique().tolist())
            
    st.markdown(body = '***')
    
    
    mühərrik_hecmi, mühərrik_gucu = st.columns(spec = [1, 1])
    
    with mühərrik_hecmi:
        mühərrik_hecmi = st.number_input(label = 'Engine volume (cm³)', value = 0, step = 50 )
        button_text = 'Send values'
    
    with mühərrik_gucu:
        mühərrik_gucu = st.number_input(label = 'Engine power(a.g.)', value = 0.0, step = 1.0, format="%.1f" )
    
    st.markdown(body = '***') 
    
    
    avtosalon,yeni = st.columns(spec = [1, 1])
    
    with avtosalon:
        avtosalon = st.checkbox(label = 'Avtosalon')
        
    with yeni:
        yeni = st.checkbox(label = 'New')
        
        
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)

     
    st.subheader(body = 'Condition')
    
    rənglənib, vuruğu_var = st.columns(spec = [1, 1])
    
    with rənglənib:
        rənglənib = st.radio(label = 'Is it colored? ', options = ['rənglənib', 'rənglənməyib'], horizontal = True)
        
    with vuruğu_var:
        vuruğu_var = st.radio(label = 'Does it have a stroke?', options = ['vuruğu var', 'vuruğu yoxdur'], horizontal = True)
    
    
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
        
   
    st.subheader(body = 'Car supply')
    
    
    lehimli_disk, abs_, lyuk, yağış_sensoru,dəri_salon = st.columns(spec = [1, 1, 1, 1, 1])
   
    with lehimli_disk:
        lehimli_disk = st.checkbox(label = 'Solder disc')
        
    with abs_:
        abs_ = st.checkbox(label = 'ABS')
    
    with lyuk:
        lyuk = st.checkbox(label = 'Lyuk')
        
    with yağış_sensoru:
        yağış_sensoru = st.checkbox(label = 'Rain sensor')
        
    with dəri_salon:
        dəri_salon = st.checkbox(label = 'Skin salon')
        
    
    st.markdown(body = '***')
 
        
    
    mərkəzi_qapanma,park_radarı, kondisioner, oturacaqların_isidilməsi,  = st.columns(spec = [1, 1, 1, 1])
    
    with mərkəzi_qapanma:
        mərkəzi_qapanma = st.checkbox(label = 'Central locking')
    
    with park_radarı:
        park_radarı = st.checkbox(label = 'Parking radar')
        
    with kondisioner:
        kondisioner = st.checkbox(label = 'Air conditioning')
    
    with oturacaqların_isidilməsi:
        oturacaqların_isidilməsi = st.checkbox(label = 'Heated seats')
        
        
    st.markdown(body = '***')
    
    
    ksenon_lampalar, arxa_görüntü_kamerası, yan_pərdələr, oturacaqların_ventilyasiyası = st.columns(spec = [1, 1, 1, 1])
    
    with ksenon_lampalar:
        ksenon_lampalar = st.checkbox(label = 'Xenon lamps')
    
    with arxa_görüntü_kamerası:
        arxa_görüntü_kamerası = st.checkbox(label = 'Rear view camera')
        
    with yan_pərdələr:
        yan_pərdələr = st.checkbox(label = 'Side curtains')
    
    with oturacaqların_ventilyasiyası:
        oturacaqların_ventilyasiyası = st.checkbox(label = 'Seat ventilation')
        
        
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)


    rənglənib_encoding = {'rənglənməyib':1,'rənglənib':0}
    vuruğu_var_encoding = {'vuruğu yoxdur':1,'vuruğu var':0}
       
    
    df['marka'] = marka_encoding
    df['model'] = model_encoding
    df['şəhər'] = şəhər_encoding
    df['yanacaq_novu'] = yanacaq_novu_encoding
    df['ötürücü'] = ötürücü_encoding
    df['ban_növü'] = ban_növü_encoding
    df['sürətlər_qutusu'] = sürətlər_qutusu_encoding
    df['rəng'] = rəng_encoding
    df['hansı_bazar_üçün_yığılıb'] = hansı_bazar_encoding
    df['rənglənib'] = df['rənglənib'].replace(rənglənib_encoding)
    df['vuruğu_var'] = df['vuruğu_var'].replace(vuruğu_var_encoding)
    


    marka = marka_mapping[marka]
    model = model_mapping[model]
    şəhər = şəhər_mapping[şəhər]
    yanacaq_novu = yanacaq_novu_mapping[yanacaq_novu]
    ötürücü = ötürücü_mapping[ötürücü]
    ban_növü = ban_növü_mapping[ban_növü]
    sürətlər_qutusu = sürətlər_qutusu_mapping[sürətlər_qutusu]
    rəng = rəng_mapping[rəng]
    hansı_bazar_üçün_yığılıb = hansı_bazar_mapping[hansı_bazar_üçün_yığılıb]
    rənglənib = rənglənib_encoding[rənglənib]
    vuruğu_var = vuruğu_var_encoding[vuruğu_var]
    
    lehimli_disk = int(lehimli_disk)
    abs_ = int(lehimli_disk)
    lyuk = int(lyuk)
    yağış_sensoru = int(yağış_sensoru)
    mərkəzi_qapanma = int(mərkəzi_qapanma)
    park_radarı = int(park_radarı)
    kondisioner = int(kondisioner)
    oturacaqların_isidilməsi = int(oturacaqların_isidilməsi)
    dəri_salon = int(dəri_salon)
    ksenon_lampalar = int(ksenon_lampalar)
    arxa_görüntü_kamerası = int(arxa_görüntü_kamerası)
    yan_pərdələr = int(yan_pərdələr)
    oturacaqların_ventilyasiyası = int(oturacaqların_ventilyasiyası)
    avtosalon = int(avtosalon)
    yeni = int(yeni)
       

    
    input_features = pd.DataFrame({
        'avtosalon': [avtosalon],
        'şəhər':[şəhər],
        'marka': [marka],
        'model': [model],
        'ban_növü': [ban_növü],
        'rəng': [rəng],
        'sürətlər_qutusu': [sürətlər_qutusu],
        'ötürücü': [ötürücü],
        'yeni': [yeni],
        'hansı_bazar_üçün_yığılıb': [hansı_bazar_üçün_yığılıb],
        'yanacaq_novu': [yanacaq_novu],
        'vuruğu_var': [vuruğu_var],
        'rənglənib': [rənglənib],
        'lehimli_disk': [lehimli_disk],
        'abs_': [abs_],
        'lyuk': [lyuk],
        'yağış_sensoru': [yağış_sensoru],
        'mərkəzi_qapanma': [mərkəzi_qapanma],
        'park_radarı': [park_radarı],
        'kondisioner': [kondisioner],
        'oturacaqların_isidilməsi': [oturacaqların_isidilməsi],
        'dəri_salon': [dəri_salon],
        'ksenon_lampalar': [ksenon_lampalar],
        'arxa_görüntü_kamerası': [arxa_görüntü_kamerası],
        'yan_pərdələr': [yan_pərdələr],
        'oturacaqların_ventilyasiyası': [oturacaqların_ventilyasiyası],
        'buraxılış_ili': [buraxılış_ili],
        'yürüş': [yürüş],
        'mühərrik_hecmi': [mühərrik_hecmi],
        'mühərrik_gucu': [mühərrik_gucu]
        
    })
    
    

    st.subheader(body = 'Model Prediction')
    
    with open('car_model.pickle', 'rb') as pickled_model:
        
        model = pickle.load(pickled_model)
    
    if st.button('Predict'):
        cars_price = model.predict(input_features)

        with st.spinner('Sending input features to model...'):
            time.sleep(2)

        st.success('Prediction is ready')
        time.sleep(1)
        st.markdown(f'### Car\'s estimated price is:  {cars_price} AZN')
        
