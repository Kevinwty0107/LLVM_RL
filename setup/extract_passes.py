import re
from subprocess import check_output
from config.config import PASS_LIST_LOC

out = check_output(['opt', '-print-passes'])
lines = out.decode('utf-8').split('\n')
analysis = False

analysis_passes = []
transform_passes = []
printer_passes = []
sanitizer_passes = []

for line in lines:
    m = re.match(r'^  ([\S]*)', line)
    if m:
        p = m.group(1)
        p = re.sub(r'<[^<>]*>', '', p)
        print(p)
        if p.find('print') >= 0:
            printer_passes.append(p)
        elif p.find('san') >= 0:
            sanitizer_passes.append(p)
        elif analysis:
            analysis_passes.append(p)
        else:
            transform_passes.append(p)
    else:
        m = re.search('analyses', line, re.IGNORECASE)
        if m:
            print('===========analysis==========')
            analysis = True
        else:
            print('==========transform==========')
            analysis = False

analysis_passes, transform_passes, printer_passes, sanitizer_passes = [list(set(passes)) for passes in
                                                                       [analysis_passes, transform_passes,
                                                                        printer_passes, sanitizer_passes]]

with open(PASS_LIST_LOC, 'w+') as f:
    f.write(f'ANALYSIS_PASSES = {analysis_passes}\n\n')
    f.write(f'TRANSFORM_PASSES = {transform_passes}\n\n')
    f.write(f'PRINTER_PASSES = {printer_passes}\n\n')
    f.write(f'SANITIZER_PASSES = {sanitizer_passes}\n\n')
    f.write(f'ALL_PASSES = {analysis_passes + transform_passes}\n\n')
