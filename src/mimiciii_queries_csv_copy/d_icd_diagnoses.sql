copy (
select * from mimiciii.d_icd_diagnoses
) to '/mimiciii_query_results/d_icd_diagnoses.csv' with delimiter ',' csv header;