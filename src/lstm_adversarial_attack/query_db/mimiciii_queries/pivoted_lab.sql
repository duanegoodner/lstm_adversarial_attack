-- create a table which has fuzzy boundaries on ICU admission (+- 12 hours from documented time)
-- this is used to assign icustay_id to lab data, which can be collected outside ICU
-- involves first creating a lag/lead version of intime/outtime
with i as (
	select subject_id,
		icustay_id,
		intime,
		outtime,
		lag (outtime) over (
			partition by subject_id
			order by intime
		) as outtime_lag,
		lead (intime) over (
			partition by subject_id
			order by intime
		) as intime_lead
	from mimiciii.icustays
),
i_vsep as (
	select subject_id,
		icustay_id,
		intime,
		outtime,
		(intime - outtime_lag) as time_since_last_discharge,
		(intime_lead - outtime) as time_until_next_admission
	from i
),
iid_assign as (
	select i_vsep.subject_id,
		i_vsep.icustay_id -- this rule is:
		--  if there are two ICU stays within 24 hours, set the start/stop
		--  time as half way between the two ICU stays
,
		case
			when i_vsep.time_since_last_discharge is not null
			and i_vsep.time_since_last_discharge < (interval '24 hours') then i_vsep.intime - (i_vsep.time_since_last_discharge) / 2
			else (i_vsep.intime - interval '12 hours')
		end as data_start,
		case
			when i_vsep.time_until_next_admission is not null
			and i_vsep.time_until_next_admission < (interval '24 hours') then i_vsep.outtime + (i_vsep.time_until_next_admission) / 2
			else (i_vsep.outtime + interval '12 hours')
		end as data_end
	from i_vsep
),
-- also create fuzzy boundaries on hospitalization
h as (
	select subject_id,
		hadm_id,
		admittime,
		dischtime,
		lag (dischtime) over (
			partition by subject_id
			order by admittime
		) as dischtime_lag,
		lead (admittime) over (
			partition by subject_id
			order by admittime
		) as admittime_lead
	from mimiciii.admissions
),
h_vsep as (
	select subject_id,
		hadm_id,
		admittime,
		dischtime,
		(admittime - dischtime_lag) as time_since_last_discharge,
		(admittime_lead - dischtime) as time_until_next_admission
	from h
),
adm as (
	select h_vsep.subject_id,
		h_vsep.hadm_id,
		-- this rule is:
		--  if there are two hospitalizations within 24 hours, set the start/stop
		--  time as half way between the two admissions
		case
			when h_vsep.time_since_last_discharge is not null
			and h_vsep.time_since_last_discharge < (interval '24 hours') then h_vsep.admittime - (h_vsep.time_since_last_discharge) / 2
			else (h_vsep.admittime - interval '12 hours')
		end as data_start,
		case
			when h_vsep.time_until_next_admission is not null
			and h_vsep.time_until_next_admission < (interval '24 hours') then h_vsep.dischtime + (h_vsep.time_until_next_admission) / 2
			else (h_vsep.dischtime + interval '12 hours')
		end as data_end
	from h_vsep
),
le_avg as (
	select pvt.subject_id,
		pvt.charttime,
		avg(
			case
				when label = 'ANION GAP' then valuenum
				else null
			end
		) as ANIONGAP,
		avg(
			case
				when label = 'ALBUMIN' then valuenum
				else null
			end
		) as ALBUMIN,
		avg(
			case
				when label = 'BANDS' then valuenum
				else null
			end
		) as BANDS,
		avg(
			case
				when label = 'BICARBONATE' then valuenum
				else null
			end
		) as BICARBONATE,
		avg(
			case
				when label = 'BILIRUBIN' then valuenum
				else null
			end
		) as BILIRUBIN,
		avg(
			case
				when label = 'CREATININE' then valuenum
				else null
			end
		) as CREATININE,
		avg(
			case
				when label = 'CHLORIDE' then valuenum
				else null
			end
		) as CHLORIDE,
		avg(
			case
				when label = 'GLUCOSE' then valuenum
				else null
			end
		) as GLUCOSE,
		avg(
			case
				when label = 'HEMATOCRIT' then valuenum
				else null
			end
		) as HEMATOCRIT,
		avg(
			case
				when label = 'HEMOGLOBIN' then valuenum
				else null
			end
		) as HEMOGLOBIN,
		avg(
			case
				when label = 'LACTATE' then valuenum
				else null
			end
		) as LACTATE,
		avg(
			case
				when label = 'MAGNESIUM' then valuenum
				else null
			end
		) as MAGNESIUM,
		avg(
			case
				when label = 'PLATELET' then valuenum
				else null
			end
		) as PLATELET,
		avg(
			case
				when label = 'POTASSIUM' then valuenum
				else null
			end
		) as POTASSIUM,
		avg(
			case
				when label = 'PTT' then valuenum
				else null
			end
		) as PTT,
		avg(
			case
				when label = 'INR' then valuenum
				else null
			end
		) as INR,
		avg(
			case
				when label = 'PT' then valuenum
				else null
			end
		) as PT,
		avg(
			case
				when label = 'SODIUM' then valuenum
				else null
			end
		) as SODIUM,
		avg(
			case
				when label = 'BUN' then valuenum
				else null
			end
		) as BUN,
		avg(
			case
				when label = 'WBC' then valuenum
				else null
			end
		) as WBC
	from (
			-- begin query that extracts the data
			select le.subject_id,
				le.hadm_id,
				le.charttime -- here we assign labels to ITEMIDs
				-- this also fuses together multiple ITEMIDs containing the same data
,
				case
					when itemid = 50868 then 'ANION GAP'
					when itemid = 50862 then 'ALBUMIN'
					when itemid = 51144 then 'BANDS'
					when itemid = 50882 then 'BICARBONATE'
					when itemid = 50885 then 'BILIRUBIN'
					when itemid = 50912 then 'CREATININE' -- exclude blood gas
					-- WHEN itemid = 50806 THEN 'CHLORIDE'
					when itemid = 50902 then 'CHLORIDE' -- exclude blood gas
					-- WHEN itemid = 50809 THEN 'GLUCOSE'
					when itemid = 50931 then 'GLUCOSE' -- exclude blood gas
					--WHEN itemid = 50810 THEN 'HEMATOCRIT'
					when itemid = 51221 then 'HEMATOCRIT' -- exclude blood gas
					--WHEN itemid = 50811 THEN 'HEMOGLOBIN'
					when itemid = 51222 then 'HEMOGLOBIN'
					when itemid = 50813 then 'LACTATE'
					when itemid = 51265 then 'PLATELET' -- exclude blood gas
					-- WHEN itemid = 50822 THEN 'POTASSIUM'
					when itemid = 50971 then 'POTASSIUM'
					when itemid = 51275 then 'PTT'
					when itemid = 51237 then 'INR'
					when itemid = 51274 then 'PT' -- exclude blood gas
					-- WHEN itemid = 50824 THEN 'SODIUM'
					when itemid = 50983 then 'SODIUM'
					when itemid = 51006 then 'BUN'
					when itemid = 51300 then 'WBC'
					when itemid = 51301 then 'WBC'
					when itemid = 50960 then 'MAGNESIUM'
					else null
				end as label,
				-- add in some sanity checks on the values
				-- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
				case
					when itemid = 50862
					and valuenum > 10 then null -- g/dL 'ALBUMIN'
					when itemid = 50868
					and valuenum > 10000 then null -- mEq/L 'ANION GAP'
					when itemid = 51144
					and valuenum < 0 then null -- immature band forms, %
					when itemid = 51144
					and valuenum > 100 then null -- immature band forms, %
					when itemid = 50882
					and valuenum > 10000 then null -- mEq/L 'BICARBONATE'
					when itemid = 50885
					and valuenum > 150 then null -- mg/dL 'BILIRUBIN'
					when itemid = 50806
					and valuenum > 10000 then null -- mEq/L 'CHLORIDE'
					when itemid = 50902
					and valuenum > 10000 then null -- mEq/L 'CHLORIDE'
					when itemid = 50912
					and valuenum > 150 then null -- mg/dL 'CREATININE'
					when itemid = 50809
					and valuenum > 10000 then null -- mg/dL 'GLUCOSE'
					when itemid = 50931
					and valuenum > 10000 then null -- mg/dL 'GLUCOSE'
					when itemid = 50810
					and valuenum > 100 then null -- % 'HEMATOCRIT'
					when itemid = 51221
					and valuenum > 100 then null -- % 'HEMATOCRIT'
					when itemid = 50811
					and valuenum > 50 then null -- g/dL 'HEMOGLOBIN'
					when itemid = 51222
					and valuenum > 50 then null -- g/dL 'HEMOGLOBIN'
					when itemid = 50813
					and valuenum > 50 then null -- mmol/L 'LACTATE'
					when itemid = 51265
					and valuenum > 10000 then null -- K/uL 'PLATELET'
					when itemid = 50822
					and valuenum > 30 then null -- mEq/L 'POTASSIUM'
					when itemid = 50971
					and valuenum > 30 then null -- mEq/L 'POTASSIUM'
					when itemid = 51275
					and valuenum > 150 then null -- sec 'PTT'
					when itemid = 51237
					and valuenum > 50 then null -- 'INR'
					when itemid = 51274
					and valuenum > 150 then null -- sec 'PT'
					when itemid = 50824
					and valuenum > 200 then null -- mEq/L == mmol/L 'SODIUM'
					when itemid = 50983
					and valuenum > 200 then null -- mEq/L == mmol/L 'SODIUM'
					when itemid = 51006
					and valuenum > 300 then null -- 'BUN'
					when itemid = 51300
					and valuenum > 1000 then null -- 'WBC'
					when itemid = 51301
					and valuenum > 1000 then null -- 'WBC'
					else valuenum
				end as valuenum
			from mimiciii.labevents le
			where le.ITEMID in (
					-- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
					50868,
					-- ANION GAP | CHEMISTRY | BLOOD | 769895
					50862,
					-- ALBUMIN | CHEMISTRY | BLOOD | 146697
					51144,
					-- BANDS - hematology
					50882,
					-- BICARBONATE | CHEMISTRY | BLOOD | 780733
					50885,
					-- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
					50912,
					-- CREATININE | CHEMISTRY | BLOOD | 797476
					50902,
					-- CHLORIDE | CHEMISTRY | BLOOD | 795568
					-- 50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
					50931,
					-- GLUCOSE | CHEMISTRY | BLOOD | 748981
					-- 50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
					51221,
					-- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
					-- 50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
					51222,
					-- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
					-- 50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
					50813,
					-- LACTATE | BLOOD GAS | BLOOD | 187124
					51265,
					-- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
					50971,
					-- POTASSIUM | CHEMISTRY | BLOOD | 845825
					-- 50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
					51275,
					-- PTT | HEMATOLOGY | BLOOD | 474937
					51237,
					-- INR(PT) | HEMATOLOGY | BLOOD | 471183
					51274,
					-- PT | HEMATOLOGY | BLOOD | 469090
					50983,
					-- SODIUM | CHEMISTRY | BLOOD | 808489
					-- 50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
					51006,
					-- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
					51301,
					-- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
					51300, -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
					50960
				)
				and valuenum is not null
				and valuenum > 0 -- lab values cannot be 0 and cannot be negative
		) pvt
	group by pvt.subject_id,
		pvt.charttime
)
select iid.icustay_id,
	adm.hadm_id,
	le_avg.*
from le_avg
	left join adm on le_avg.subject_id = adm.subject_id
	and le_avg.charttime >= adm.data_start
	and le_avg.charttime < adm.data_end
	left join iid_assign iid on le_avg.subject_id = iid.subject_id
	and le_avg.charttime >= iid.data_start
	and le_avg.charttime < iid.data_end
order by le_avg.subject_id,
	le_avg.charttime;