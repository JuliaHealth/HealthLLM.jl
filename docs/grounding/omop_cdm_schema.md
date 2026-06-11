# OMOP CDM Schema Reference (Common Tables)

## person
- `person_id` INTEGER PRIMARY KEY
- `gender_concept_id` INTEGER (FK → concept)
- `year_of_birth` INTEGER
- `month_of_birth` INTEGER
- `day_of_birth` INTEGER
- `birth_datetime` DATETIME
- `race_concept_id` INTEGER
- `ethnicity_concept_id` INTEGER
- `location_id` INTEGER
- `provider_id` INTEGER
- `care_site_id` INTEGER (FK → care_site)
- `person_source_value` VARCHAR
- `gender_source_value` VARCHAR
- `race_source_value` VARCHAR
- `ethnicity_source_value` VARCHAR

## observation_period
- `observation_period_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `observation_period_start_date` DATE
- `observation_period_end_date` DATE
- `period_type_concept_id` INTEGER

## visit_occurrence
- `visit_occurrence_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `visit_concept_id` INTEGER (FK → concept)
- `visit_start_date` DATE
- `visit_start_datetime` DATETIME
- `visit_end_date` DATE
- `visit_end_datetime` DATETIME
- `visit_type_concept_id` INTEGER
- `provider_id` INTEGER
- `care_site_id` INTEGER
- `visit_source_value` VARCHAR
- `visit_source_concept_id` INTEGER

## condition_occurrence
- `condition_occurrence_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `condition_concept_id` INTEGER (FK → concept)
- `condition_start_date` DATE
- `condition_start_datetime` DATETIME
- `condition_end_date` DATE
- `condition_end_datetime` DATETIME
- `condition_type_concept_id` INTEGER
- `condition_status_concept_id` INTEGER
- `stop_reason` VARCHAR
- `provider_id` INTEGER
- `visit_occurrence_id` INTEGER
- `condition_source_value` VARCHAR
- `condition_source_concept_id` INTEGER

## condition_era
- `condition_era_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `condition_concept_id` INTEGER (FK → concept)
- `condition_era_start_date` DATE
- `condition_era_end_date` DATE
- `condition_occurrence_count` INTEGER

## drug_exposure
- `drug_exposure_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `drug_concept_id` INTEGER (FK → concept)
- `drug_exposure_start_date` DATE
- `drug_exposure_start_datetime` DATETIME
- `drug_exposure_end_date` DATE
- `drug_exposure_end_datetime` DATETIME
- `verbatim_end_date` DATE
- `drug_type_concept_id` INTEGER
- `stop_reason` VARCHAR
- `refills` INTEGER
- `quantity` NUMERIC
- `days_supply` INTEGER
- `sig` VARCHAR
- `route_concept_id` INTEGER
- `lot_number` VARCHAR
- `provider_id` INTEGER
- `visit_occurrence_id` INTEGER
- `drug_source_value` VARCHAR
- `drug_source_concept_id` INTEGER

## drug_era
- `drug_era_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `drug_concept_id` INTEGER (FK → concept)
- `drug_era_start_date` DATE
- `drug_era_end_date` DATE
- `drug_occurrence_count` INTEGER

## procedure_occurrence
- `procedure_occurrence_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `procedure_concept_id` INTEGER (FK → concept)
- `procedure_date` DATE
- `procedure_datetime` DATETIME
- `procedure_end_date` DATE
- `procedure_end_datetime` DATETIME
- `procedure_type_concept_id` INTEGER
- `modifier_concept_id` INTEGER
- `quantity` INTEGER
- `provider_id` INTEGER
- `visit_occurrence_id` INTEGER
- `procedure_source_value` VARCHAR
- `procedure_source_concept_id` INTEGER

## observation
- `observation_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `observation_concept_id` INTEGER (FK → concept)
- `observation_date` DATE
- `observation_datetime` DATETIME
- `observation_type_concept_id` INTEGER
- `value_as_number` NUMERIC
- `value_as_string` VARCHAR
- `value_as_concept_id` INTEGER
- `qualifier_concept_id` INTEGER
- `unit_concept_id` INTEGER
- `provider_id` INTEGER
- `visit_occurrence_id` INTEGER
- `observation_source_value` VARCHAR
- `observation_source_concept_id` INTEGER

## measurement
- `measurement_id` INTEGER PRIMARY KEY
- `person_id` INTEGER (FK → person)
- `measurement_concept_id` INTEGER (FK → concept)
- `measurement_date` DATE
- `measurement_datetime` DATETIME
- `measurement_type_concept_id` INTEGER
- `operator_concept_id` INTEGER
- `value_as_number` NUMERIC
- `value_as_concept_id` INTEGER
- `unit_concept_id` INTEGER
- `range_low` NUMERIC
- `range_high` NUMERIC
- `provider_id` INTEGER
- `visit_occurrence_id` INTEGER
- `measurement_source_value` VARCHAR
- `measurement_source_concept_id` INTEGER

## death
- `person_id` INTEGER PRIMARY KEY (FK → person)
- `death_date` DATE
- `death_datetime` DATETIME
- `death_type_concept_id` INTEGER
- `cause_concept_id` INTEGER
- `cause_source_value` VARCHAR
- `cause_source_concept_id` INTEGER

## care_site
- `care_site_id` INTEGER PRIMARY KEY
- `care_site_name` VARCHAR
- `place_of_service_concept_id` INTEGER (FK → concept)
- `location_id` INTEGER
- `care_site_source_value` VARCHAR
- `place_of_service_source_value` VARCHAR

## provider
- `provider_id` INTEGER PRIMARY KEY
- `provider_name` VARCHAR
- `npi` VARCHAR
- `dea` VARCHAR
- `specialty_concept_id` INTEGER
- `care_site_id` INTEGER
- `year_of_birth` INTEGER
- `gender_concept_id` INTEGER
- `provider_source_value` VARCHAR
- `specialty_source_value` VARCHAR
- `specialty_source_concept_id` INTEGER
- `gender_source_value` VARCHAR
- `gender_source_concept_id` INTEGER

## concept
- `concept_id` INTEGER PRIMARY KEY
- `concept_name` VARCHAR(255)
- `domain_id` VARCHAR(20)
- `vocabulary_id` VARCHAR(20)
- `concept_class_id` VARCHAR(20)
- `standard_concept` VARCHAR(1)
- `concept_code` VARCHAR(50)
- `valid_start_date` DATE
- `valid_end_date` DATE
- `invalid_reason` VARCHAR(1)

## concept_ancestor
- `ancestor_concept_id` INTEGER
- `descendant_concept_id` INTEGER
- `min_levels_of_separation` INTEGER
- `max_levels_of_separation` INTEGER

## concept_relationship
- `concept_id_1` INTEGER
- `concept_id_2` INTEGER
- `relationship_id` VARCHAR(20)
- `valid_start_date` DATE
- `valid_end_date` DATE
- `invalid_reason` VARCHAR(1)

## concept_synonym
- `concept_id` INTEGER
- `concept_synonym_name` VARCHAR(1000)
- `language_concept_id` INTEGER

## location
- `location_id` INTEGER PRIMARY KEY
- `address_1` VARCHAR(50)
- `address_2` VARCHAR(50)
- `city` VARCHAR(50)
- `state` VARCHAR(2)
- `zip` VARCHAR(9)
- `county` VARCHAR(20)
- `location_source_value` VARCHAR(50)

## Common Query Patterns in OMOP

1. **Count patients by gender**: Join person → concept (for gender_concept_id)
2. **Find patients with a condition**: Join condition_era → concept_ancestor (for SNOMED hierarchy)
3. **Compute length of stay**: DATEDIFF on visit_start/end_date
4. **Find comorbidities**: Subqueries finding patients with condition A, then finding their other conditions
5. **Temporal patterns**: Filter conditions by date ranges, seasons, year/month
6. **Drug utilization**: Join drug_exposure/drug_era → concept for drug details
