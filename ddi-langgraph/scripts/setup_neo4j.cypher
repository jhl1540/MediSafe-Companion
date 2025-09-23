CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug) REQUIRE d.name IS UNIQUE;

CREATE INDEX drug_name_idx IF NOT EXISTS FOR (d:Drug) ON (d.name);
CREATE INDEX source_name_idx IF NOT EXISTS FOR (s:Source) ON (s.name);
