format:
	autoflake -i --ignore-init-module-imports --remove-all-unused-imports -r mae_lite
	black --line-length 120 .

style_check:
	autoflake --ignore-init-module-imports --remove-all-unused-imports -r mae_lite
	black --line-length 120 --diff --check .
