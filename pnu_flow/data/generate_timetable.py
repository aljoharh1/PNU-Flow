"""
Citations (open-source):
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd
from pnu_flow.config import SIM_CFG


@dataclass
class TimetableGenerator:
    seed: int = 42
    def __post_init__(self):
        random.seed(self.seed); np.random.seed(self.seed)

    def _class_slots(self) -> List[tuple]:
        return [("08:00","09:15"),("09:30","10:45"),("11:00","12:15"),
                ("12:30","13:45"),("14:00","15:15"),("15:30","16:45")]

    def generate(self, n_sections: int = 120) -> pd.DataFrame:
        rooms = ["lecture_hall_201","study_hall_1","corridor_A_2","corridor_B_2"]
        slots = self._class_slots()
        rows = []
        for i in range(n_sections):
            s,e = random.choice(slots)
            r   = random.choice(rooms)
            rows.append({"course_code":f"CAI{300+(i%90)}","section":f"S{i:03d}",
                         "room_number":r,"floor":2 if "_2" in r or "201" in r else 1,
                         "day":random.choice(SIM_CFG.days),"start_time":s,"end_time":e,
                         "enrolled_students":int(np.random.randint(18,52))})
        return pd.DataFrame(rows)


def expand_timetable_to_week_events(df: pd.DataFrame,
                                     reference_date: str = "2026-03-22") -> pd.DataFrame:
    base = datetime.fromisoformat(reference_date)
    offset = {"Sun":0,"Mon":1,"Tue":2,"Wed":3,"Thu":4}
    events = []
    for _,row in df.iterrows():
        d = base + timedelta(days=offset[row["day"]])
        ts = datetime.strptime(f"{d.date()} {row['end_time']}", "%Y-%m-%d %H:%M")
        events.append({"timestamp":ts,"course_code":row["course_code"],
                       "room_number":row["room_number"],
                       "enrolled_students":row["enrolled_students"],"day":row["day"]})
    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
