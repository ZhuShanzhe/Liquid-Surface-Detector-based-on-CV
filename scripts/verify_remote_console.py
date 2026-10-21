"""Exercise our own server-backed HTML and UI in real browser contexts."""
import argparse
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true', help='Verify entry/retry/default measurement after a transport change')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(success=False, cases=[], checks={}, javascript_errors=[], conditional_status_responses=0,
                  scope='Server-backed HTML functional integration; not a statistical accuracy evaluation')

    def snapshot(page, name):
        data = page.evaluate('''() => ({actual:ui.frame.actual,estimate:ui.frame.estimate,
           controls:ui.frame.controls,timing:ui.frame.timing,
           eligible:ui.frame.input_qualification.eligible,
           measured:ui.frame.result.inference_executed,
           initialization_id:ui.frame.initialization.id,
           rendering:ui.frame.rendering})''')
        data['name'] = name
        data['liquid_absolute_error_mm'] = (None if data['estimate']['liquid_height_m'] is None else
            abs(data['estimate']['liquid_height_m'] - data['actual']['liquid_height_m']) * 1000)
        report['cases'].append(data)
        print(json.dumps(dict(event='browser_case_complete',name=name,eligible=data['eligible'],
                             numeric=data['estimate']['liquid_height_m'] is not None)), flush=True)
        return data

    def ready(page, expected=None, previous=None):
        try:
            page.wait_for_function('''args => typeof ui!=='undefined' && ui.frame && !ui.busy &&
           (!args.previous || ui.frame.frame_id!==args.previous) &&
           (args.expected===null || Math.abs(ui.frame.controls.height_m-args.expected)<1e-7)''',
           arg=dict(expected=expected, previous=previous),timeout=60000)
        except Exception:
            report['ui_failure_state'] = page.evaluate('''() => ({busy:ui.busy,lastJob:ui.lastJob,
                frame:ui.frame?.frame_id,status:document.getElementById('status').textContent,
                message:document.getElementById('message').textContent,
                showError:window.__remoteShowError??null})''')
            page.screenshot(path=str(args.output.parent/'failure.png'),full_page=True)
            raise

    def apply(page, name, expected):
        old = page.evaluate('ui.frame.frame_id')
        page.locator('#apply').click()
        ready(page, expected, old)
        return snapshot(page, name)

    def value(page, selector, value):
        page.locator(selector).evaluate("(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));}", str(value))

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--disable-dev-shm-usage'])
            first = browser.new_context(viewport={'width':1560,'height':1120})
            first.add_init_script('''window.addEventListener('DOMContentLoaded',()=>{
              if(typeof show==='function'){const original=show;show=(...args)=>{
                try{return original(...args)}catch(error){window.__remoteShowError=String(error.stack);throw error;}
              };}
            });''')
            page = first.new_page()
            page.on('pageerror', lambda error: report['javascript_errors'].append(str(error)))

            def response_seen(response):
                if response.url.endswith('/api/status') and response.status == 304:
                    report['conditional_status_responses'] += 1

            page.on('response', response_seen)
            retried = []
            if args.quick:
                def health_route(route):
                    if not retried:
                        retried.append(True)
                        route.fulfill(status=503,body='{}',headers={'Access-Control-Allow-Origin':'*'})
                    else:
                        route.continue_()
                page.route('**/healthz', health_route)
            page.goto(args.entry.resolve().as_uri())
            page.wait_for_url('**/console',timeout=120000)
            ready(page, .6)
            initial = snapshot(page, 'default_from_transferred_html')
            report['checks']['html_opened_full_server_ui'] = page.url.startswith('https://') and page.url.endswith('/console')
            report['checks']['rgb_and_depth_loaded'] = page.evaluate("document.querySelector('#rgb').naturalWidth>0&&document.querySelector('#depth').naturalWidth>0")
            report['checks']['server_branding'] = '服务器仿真实验台' in page.title()
            report['checks']['access_key_removed_from_address'] = '#' not in page.url
            if args.quick:
                report['checks']['launcher_retries_temporary_connection_failure'] = bool(retried)
                report['checks']['qualified_default_measurement_numeric'] = initial['eligible'] and initial['measured'] and initial['estimate']['liquid_height_m'] is not None
                report['checks']['no_javascript_errors'] = not report['javascript_errors']
                report['success'] = all(report['checks'].values())
                report['scope'] = 'Post-transport-change HTML retry and default measurement smoke; injected initial health failure is not a performance sample'
                first.close()
                browser.close()
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(dict(success=report['success'],checks=report['checks']),ensure_ascii=False),flush=True)
                return 0 if report['success'] else 1
            page.locator('#height').fill('0.8')
            changed = apply(page, 'liquid_adjustment', .8)
            report['checks']['height_change_reuses_calibration'] = initial['initialization_id'] == changed['initialization_id']
            second = browser.new_context(viewport={'width':1560,'height':1120})
            other = second.new_page()
            other.goto(args.entry.resolve().as_uri())
            other.wait_for_url('**/console',timeout=120000)
            ready(other, .6)
            own = snapshot(other, 'independent_second_browser')
            report['checks']['browser_states_are_independent'] = (
                own['initialization_id'] != changed['initialization_id'] and
                page.evaluate('ui.frame.actual.liquid_height_m') == .8)
            page.locator('#tabInstallation').click()
            page.locator('#containerShape').select_option('ellipse')
            value(page, '#containerWidth', 1.2)
            value(page, '#containerDepth', .9)
            value(page, '#containerHeight', 2.)
            value(page, '#cameraBottom', 5.)
            installed = apply(page, 'ellipse_container_and_camera_change', .8)
            report['checks']['installation_change_recalibrates'] = installed['initialization_id'] != changed['initialization_id']
            page.locator('#tabOperation').click()
            for selector, v in [('#light',60),('#glare',35),('#floats',3),('#wave',20),('#loss',15)]:
                value(page, selector, v)
            compound = apply(page, 'compound_low_light_glare_floats_wave_loss', .8)
            page.screenshot(path=str(args.output.parent/'remote-compound.png'),full_page=True)
            for material in ('homogeneous_transparent','homogeneous_translucent'):
                for selector, v in [('#light',100),('#glare',0),('#floats',0),('#wave',0),('#loss',0)]:
                    value(page, selector, v)
                page.locator('#material').select_option(material)
                apply(page, material, .8)
            page.screenshot(path=str(args.output.parent/'remote-translucent.png'),full_page=True)
            report['checks']['two_images_decode_after_material_change'] = page.evaluate("document.querySelector('#rgb').naturalWidth>0&&document.querySelector('#depth').naturalWidth>0")
            # Wait for ordinary UI polling rather than issuing artificial API requests.
            page.wait_for_timeout(2500)
            report['checks']['unchanged_poll_avoids_resending_frame'] = report['conditional_status_responses'] > 0
            report['checks']['all_scenario_inputs_qualified_and_measured'] = all(r['eligible'] and r['measured'] for r in report['cases'])
            report['checks']['all_scenario_liquid_outputs_numeric'] = all(r['estimate']['liquid_height_m'] is not None for r in report['cases'])
            report['checks']['no_javascript_errors'] = not report['javascript_errors']
            report['success'] = all(report['checks'].values())
            first.close()
            second.close()
            browser.close()
    except Exception as error:
        report['failure_type'] = type(error).__name__
        # Browser exceptions may contain a launch URL; omit credential-bearing fragments.
        report['failure_message'] = str(error).split('#key=')[0][:1600]
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(success=report['success'],checks=report['checks'],cases=len(report['cases'])),ensure_ascii=False),flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
