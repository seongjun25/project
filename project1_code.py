import pandas as pd
import nycflights13 as flights

# 항공편 데이터 (main dataset)
df_flights = flights.flights
df_airlines = flights.airlines
df_airports = flights.airports
df_planes = flights.planes
df_weather = flights.weather

###1 선호시간대 분류 및 목적지 top3 출력
# 여행자의 입장에서 최적의 도착지를 선정하는 주제
# 조원들이 각기 다른 기준을 적용하며, 저는 시간대를 기준으로 top 목적지 

# 0 (자정) <= sched_dep_time < 600 (오전 6시) : "새벽"
df_flights.loc[(df_flights['sched_dep_time'] >= 0) & (df_flights['sched_dep_time'] < 600), '선호시간대'] = '새벽'

# 600 (오전 6시) <= sched_dep_time < 1200 (정오) : "전일휴가"
df_flights.loc[(df_flights['sched_dep_time'] >= 600) & (df_flights['sched_dep_time'] < 1200), '선호시간대'] = '전일휴가'

# 1200 (정오) <= sched_dep_time < 1800 (오후 6시) : "반차"
df_flights.loc[(df_flights['sched_dep_time'] >= 1200) & (df_flights['sched_dep_time'] < 1800), '선호시간대'] = '반차'

# 1800 (오후 6시) <= sched_dep_time < 2400 (다음날 자정 직전) : "퇴근후"
df_flights.loc[(df_flights['sched_dep_time'] >= 1800) & (df_flights['sched_dep_time'] < 2400), '선호시간대'] = '퇴근후'


# 2. 카테고리형 변수로 변환
df_flights['선호시간대'] = df_flights['선호시간대'].astype('category')


# 1. 결측치 제거 (arr_delay, dest, 선호시간대가 없는 행 제거)
df_dropna = df_flights.dropna(subset=['arr_delay', 'dest', '선호시간대'])

# 2. 선호시간대별, 도착지별 평균 도착지연 계산
delay_by_time_dest = (
    df_dropna
    .groupby(['선호시간대', 'dest'])['arr_delay']
    .mean()
    .reset_index(name='avg_arr_delay')
)

# 3. 항공편 수가 너무 적은것 제거(10이상만 필터링)
flight10=delay_by_time_dest[df_flights["flight"]>=10]

# 4. 선호시간대별 지연시간 낮은 도착지 Top3 추출
top3_low_delay = (
    delay_by_time_dest
    .sort_values(['선호시간대', 'avg_arr_delay'], ascending=[True, True])
    .groupby('선호시간대')
    .head(3)
)

# 5. 결과 출력
top3_low_delay


## 결과시각화
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 선호시간대 순서 지정
time_order = ['새벽', '전일휴가', '반차', '퇴근후']
top3_low_delay['선호시간대'] = pd.Categorical(top3_low_delay['선호시간대'], categories=time_order, ordered=True)

# 2x2 subplot 설정
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, time in enumerate(time_order):
    ax = axes[i]
    group = top3_low_delay[top3_low_delay['선호시간대'] == time]

    # 막대 색 설정: 가장 평균지연이 낮은 항목은 빨간색, 나머지는 회색
    min_delay = group['avg_arr_delay'].min()
    colors = ['#FF6B6B' if val == min_delay else '#CFCFCF' for val in group['avg_arr_delay']]

    # 막대그래프 그리기
    bars = ax.bar(group['dest'], group['avg_arr_delay'], color=colors)

    # 막대 위 수치 표시 (크기 확대, 단위 추가)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            yval + 0.5, 
            f'{yval:.1f}분', 
            ha='center', 
            va='bottom', 
            fontsize=11,
            color='black'
        )

    # 제목, 축 설정
    ax.set_title(time, fontsize=13)
    ax.set_xlabel('도착지')
    ax.set_ylabel('평균 도착지연 (분)')
    ax.set_ylim(group['avg_arr_delay'].min() - 5, group['avg_arr_delay'].max() + 10)

    # 점선 구분선 추가
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ax.axvline(x=-0.5, color='black', linestyle=':', linewidth=1)
    ax.axvline(x=2.5, color='black', linestyle=':', linewidth=1)

# 범례 설정
red_patch = mpatches.Patch(color='#FF6B6B', label='지연시간 최저 도착지')
gray_patch = mpatches.Patch(color='#CFCFCF', label='Top3 도착지')
fig.legend(handles=[red_patch, gray_patch], loc='lower center', ncol=2)

# 전체 제목 및 여백 조정
plt.suptitle('선호시간대별 평균 도착지연이 낮은 Top3 도착지', fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # 아래 여백도 확보
plt.show()



### 2번째분석-1 : 현재시각에서 공항별로 2시간이내 지연이 적은 목적지 추출. 분석이유는 현재 공항에 가는 상황에서 지연이 적은 목적지를 추출하고싶어서

import pandas as pd

# 예시 현재 시간: 
now = 2142

# HHMM -> 분 단위로 변환하는 함수
def hhmm_to_minutes(hhmm):
    hour = hhmm // 100
    minute = hhmm % 100
    return hour * 60 + minute

now_minutes = hhmm_to_minutes(now)
two_hours_later = now_minutes + 120

# sched_dep_time을 분 단위로 변환한 새로운 컬럼 추가
df_flights["dep_time_mins"] = df_flights["sched_dep_time"].apply(hhmm_to_minutes)

# 뉴욕 공항 리스트
nyc_airports = ['JFK', 'LGA', 'EWR']

# 2시간 이내 출발 + 뉴욕공항 출발 항공편 필터링
df_2hr_nyc = df_flights[
    (df_flights["dep_time_mins"] >= now_minutes) & 
    (df_flights["dep_time_mins"] <= two_hours_later) &
    (df_flights["origin"].isin(nyc_airports))
]

# 공항별로 지연 적은 목적지 Top3 추출
top3_dest_by_origin = (
    df_2hr_nyc
    .groupby(['origin', 'dest'])['dep_delay']
    .mean()
    .reset_index()
    .rename(columns={'dep_delay': 'avg_dep_delay'})
)

# origin별로 지연 적은 목적지 Top3만 추출
top3_dest_by_origin = (
    top3_dest_by_origin
    .sort_values(['origin', 'avg_dep_delay'])
    .groupby('origin')
    .head(3)
    .reset_index(drop=True)
)

print(top3_dest_by_origin)


### 2번째분석-2 : 현재시각에서 공항별로 2시간이내 지연이 적은 항공편 추출. 분석이유는 현재 공항에 가는 상황에서 지연이 적은 목적지를 추출하고싶어서
import pandas as pd

# 예시 현재 시간: 
now = 2142

# HHMM -> 분 단위로 변환하는 함수
def hhmm_to_minutes(hhmm):
    hour = hhmm // 100
    minute = hhmm % 100
    return hour * 60 + minute

now_minutes = hhmm_to_minutes(now)
two_hours_later = now_minutes + 120

# sched_dep_time을 분 단위로 변환한 새로운 컬럼 추가
df_flights["dep_time_mins"] = df_flights["sched_dep_time"].apply(hhmm_to_minutes)

# 뉴욕 공항 리스트
nyc_airports = ['JFK', 'LGA', 'EWR']

# 2시간 이내 출발 + 뉴욕공항 출발 항공편 필터링
df_2hr_nyc = df_flights[
    (df_flights["dep_time_mins"] >= now_minutes) & 
    (df_flights["dep_time_mins"] <= two_hours_later) &
    (df_flights["origin"].isin(nyc_airports))
]

# 공항별로 항공사별 평균 출발 지연 시간 계산
carrier_delay_by_origin = (
    df_2hr_nyc
    .groupby(['origin', 'carrier'])['dep_delay']
    .mean()
    .reset_index()
    .rename(columns={'dep_delay': 'avg_dep_delay'})
)

# 공항별 지연 적은 항공사 Top3 추출
top3_carrier_by_origin = (
    carrier_delay_by_origin
    .sort_values(['origin', 'avg_dep_delay'])
    .groupby('origin')
    .head(3)
    .reset_index(drop=True)
)

print(top3_carrier_by_origin)


## 결과시각화


## ver2-1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_top3_destinations_by_delay(df_flights, now):
    def hhmm_to_minutes(hhmm):
        hour = hhmm // 100
        minute = hhmm % 100
        return hour * 60 + minute
    
    now_minutes = hhmm_to_minutes(now)
    two_hours_later = now_minutes + 120

    if 'dep_time_mins' not in df_flights.columns:
        df_flights['dep_time_mins'] = df_flights['sched_dep_time'].apply(hhmm_to_minutes)
    
    nyc_airports = ['JFK', 'LGA', 'EWR']
    
    df_2hr_nyc = df_flights[
        (df_flights["dep_time_mins"] >= now_minutes) &
        (df_flights["dep_time_mins"] <= two_hours_later) &
        (df_flights["origin"].isin(nyc_airports))
    ].dropna(subset=['dep_delay', 'dest'])
    
    top3_dest_by_origin = (
        df_2hr_nyc
        .groupby(['origin', 'dest'])['dep_delay']
        .mean()
        .reset_index()
        .rename(columns={'dep_delay': 'avg_dep_delay'})
    )
    
    top3_dest_by_origin = (
        top3_dest_by_origin
        .sort_values(['origin', 'avg_dep_delay'])
        .groupby('origin')
        .head(3)
        .reset_index(drop=True)
    )
    
    print(top3_dest_by_origin)
    
    origins = top3_dest_by_origin['origin'].unique()
    fig, axes = plt.subplots(1, len(origins), figsize=(18, 6), sharey=True)
    
    if len(origins) == 1:
        axes = [axes]
    
    for ax, origin in zip(axes, origins):
        data = top3_dest_by_origin[top3_dest_by_origin['origin'] == origin]
        x = np.arange(len(data))
        
        bars = ax.bar(x, data['avg_dep_delay'], color='skyblue', label='평균 출발 지연 시간')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)  # x=0 축 추가
        ax.set_title(f'{origin} 출발 - 지연 적은 목적지 Top3')
        ax.set_xticks(x)
        ax.set_xticklabels(data['dest'], rotation=30)
        ax.set_ylabel('평균 출발 지연 시간 (분)')
        ax.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}분', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'현재 시간 {now} 기준 2시간 이내 출발 뉴욕공항별 지연 적은 목적지 Top3', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 호출 예시
now = 2142
plot_top3_destinations_by_delay(df_flights, now)


## ver2-2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_top3_carriers_by_delay(df_flights, now):
    # HHMM -> 분 단위 변환 함수
    def hhmm_to_minutes(hhmm):
        hour = hhmm // 100
        minute = hhmm % 100
        return hour * 60 + minute
    
    now_minutes = hhmm_to_minutes(now)
    two_hours_later = now_minutes + 120
    
    # 분 단위 컬럼 추가 (필요시)
    if 'dep_time_mins' not in df_flights.columns:
        df_flights['dep_time_mins'] = df_flights['sched_dep_time'].apply(hhmm_to_minutes)
    
    nyc_airports = ['JFK', 'LGA', 'EWR']
    
    # 2시간 이내 출발, 뉴욕공항 필터링
    df_2hr_nyc = df_flights[
        (df_flights['dep_time_mins'] >= now_minutes) &
        (df_flights['dep_time_mins'] <= two_hours_later) &
        (df_flights['origin'].isin(nyc_airports))
    ].dropna(subset=['dep_delay', 'carrier'])
    
    # 공항별-항공사별 평균 지연시간 계산
    carrier_delay_by_origin = (
        df_2hr_nyc.groupby(['origin', 'carrier'])['dep_delay']
        .mean()
        .reset_index()
        .rename(columns={'dep_delay': 'avg_dep_delay'})
    )
    
    # 공항별 지연 적은 항공사 Top3 추출
    top3_carrier_by_origin = (
        carrier_delay_by_origin
        .sort_values(['origin', 'avg_dep_delay'])
        .groupby('origin')
        .head(3)
        .reset_index(drop=True)
    )
    
    print(top3_carrier_by_origin)
    
    origins = top3_carrier_by_origin['origin'].unique()
    fig, axes = plt.subplots(1, len(origins), figsize=(15, 5), sharey=True)
    if len(origins) == 1:
        axes = [axes]
    
    for ax, origin in zip(axes, origins):
        data = top3_carrier_by_origin[top3_carrier_by_origin['origin'] == origin]
        x = np.arange(len(data))
        
        bars = ax.bar(x, data['avg_dep_delay'], color='skyblue', label='평균 출발 지연 시간')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)  # y=0 기준선
        
        ax.set_title(f"{origin} 공항 - 지연 적은 항공사 Top3")
        ax.set_xticks(x)
        ax.set_xticklabels(data['carrier'], rotation=30)
        ax.set_ylabel("평균 출발 지연 시간 (분)")
        ax.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(round(height))}분", 
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f"{now} 기준 2시간 이내 출발 뉴욕공항별 지연 적은 항공사 Top3")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 사용 예시
now = 2142
plot_top3_carriers_by_delay(df_flights, now)
