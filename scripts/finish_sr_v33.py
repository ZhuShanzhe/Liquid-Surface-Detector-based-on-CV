import json
import experiment_sr_v33 as run

rows=json.loads((run.OUT/'development-expanded-raw.json').read_text())['records']
run.select(rows)
run.run_test()
run.sr_audit()
import audit_sr_v33
audit_sr_v33.audit()
