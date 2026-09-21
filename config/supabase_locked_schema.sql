-- GEOARGUS locked schema. Apply through the backend migration workflow.
create table if not exists public.site_config (
 id bigint generated always as identity primary key, site_id text unique not null, mine_name text,
 mine_type text not null check (mine_type in ('longwall','bord_and_pillar')),
 aoi_latitude double precision, aoi_longitude double precision, panel_boundary jsonb,
 depth_m double precision, extraction_thickness_m double precision, extraction_method text,
 panel_status text check (panel_status in ('completed','active','planned')), current_face_position jsonb, percent_complete double precision,
 pillar_width_m double precision, gallery_width_m double precision, seam_thickness_m double precision,
 rock_to_soil_ratio double precision, brittleness_index double precision, rock_density double precision,
 current_extraction_pct double precision, pillar_configuration jsonb, goaf_development text,
 is_assumed_geometry boolean default true, is_assumed_geology boolean default true,
 last_updated_mining_state timestamptz, emergency_contacts jsonb, created_at timestamptz default now()
);
create table if not exists public.nodes (
 id bigint generated always as identity primary key, site_id text references public.site_config(site_id), node_id integer not null,
 latitude double precision, longitude double precision, elevation_m double precision, installed_at timestamptz,
 baseline_tilt_x double precision, baseline_tilt_y double precision, baseline_displacement_mm double precision,
 source text not null check (source in ('real','mock')), unique(site_id,node_id)
);
create table if not exists public.insar_grid_features (
 id bigint generated always as identity primary key, grid_id text not null, site_id text references public.site_config(site_id),
 latitude double precision, longitude double precision, date_a date, date_b date, los_displacement_mm double precision,
 coherence double precision, coherence_class text, reference_point_latitude double precision, reference_point_longitude double precision,
 displacement_reference text default 'relative_to_insar_reference_point', product_type text default 'pair_displacement',
 orbit_direction text, created_at timestamptz default now()
);
create table if not exists public.twin_state (
 id bigint generated always as identity primary key, site_id text references public.site_config(site_id), node_id integer, grid_id text,
 mode text check (mode in ('longwall','bord_and_pillar')), expected_deformation_mm double precision,
 susceptibility_score double precision, structural_risk_indicator double precision, observed_value_mm double precision,
 physics_deviation_index double precision check (physics_deviation_index between 0 and 1), uncertainty_band_mm double precision,
 parameter_basis jsonb, is_hypothetical boolean not null default false, computed_at timestamptz default now()
);
alter table public.predictions add column if not exists panel_id text,
 add column if not exists risk_score double precision check (risk_score between 0 and 1),
 add column if not exists risk_level text check (risk_level in ('normal','watch','warning','critical')),
 add column if not exists confidence_badge text check (confidence_badge in ('agreeing','mixed','disagreeing'));
create table if not exists public.alert_feedback (
 id bigint generated always as identity primary key, prediction_id bigint references public.predictions(id), cluster_event_id bigint,
 confirmed_outcome text check (confirmed_outcome in ('real_event','false_alarm','unconfirmed')),
 notes text, confirmed_by text, confirmed_at timestamptz default now()
);
create table if not exists public.alerts (
 id bigint generated always as identity primary key, site_id text references public.site_config(site_id), prediction_id bigint references public.predictions(id),
 channel text check (channel in ('whatsapp','sms','email','voice_dtmf','dashboard')), escalation_step integer,
 contact_name text, confirmed boolean, confirmed_at timestamptz, delivery_status text, created_at timestamptz default now()
);
