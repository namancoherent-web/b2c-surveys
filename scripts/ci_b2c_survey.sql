-- ci_b2c_survey — B2C consumer surveys for the CMI platform.
--
-- Mirrors public.ci_b2b_survey exactly (same columns, types, constraints and
-- defaults), so anything already written against the B2B table works here with
-- only the table name changed.
--
-- Keying follows the established B2B convention: ONE ROW PER (region × market),
-- identified by a unique geo-prefixed snake_case market_slug, e.g.
--     global_smartwatches_market
--     europe_smartwatches_market
--     north_america_running_shoes_market
--
-- Run against the CMI Neon database.

CREATE TABLE IF NOT EXISTS public.ci_b2c_survey (
    id           serial                      PRIMARY KEY,
    market_slug  text                        NOT NULL UNIQUE,
    market_name  text                        NOT NULL,
    survey       jsonb                       NOT NULL,
    generated_on date,
    created_at   timestamp with time zone    NOT NULL DEFAULT now(),
    updated_at   timestamp with time zone    NOT NULL DEFAULT now()
);

-- Keep updated_at honest on re-publish.
CREATE OR REPLACE FUNCTION public.ci_b2c_survey_touch_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ci_b2c_survey_set_updated_at ON public.ci_b2c_survey;
CREATE TRIGGER ci_b2c_survey_set_updated_at
    BEFORE UPDATE ON public.ci_b2c_survey
    FOR EACH ROW EXECUTE FUNCTION public.ci_b2c_survey_touch_updated_at();

-- Lookups the frontend will need: by market, and "all regions of one market".
CREATE INDEX IF NOT EXISTS ci_b2c_survey_market_name_idx
    ON public.ci_b2c_survey (market_name);

-- Question-level search inside the payload (jsonb containment / path ops).
CREATE INDEX IF NOT EXISTS ci_b2c_survey_survey_gin_idx
    ON public.ci_b2c_survey USING gin (survey jsonb_path_ops);
