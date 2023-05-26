copy (
    select *
    from mimiciii.admissions
) to '/mimiciii_query_results/admissions.csv' with delimiter ',' csv header;