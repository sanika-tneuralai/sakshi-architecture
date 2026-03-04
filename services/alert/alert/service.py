"""
Alert service for processing and sending alerts.
"""
from alert.schemas import AlertRequest, AlertResponse, PipelineAlertRequest, PipelineAlertResponse, AlertDetail


def process_pipeline_alerts(request: PipelineAlertRequest) -> PipelineAlertResponse:
    """
    Process multiple usecase results and send appropriate alerts.
    
    Alert Rules:
    - person_in_roi: Send alert if triggered=True
    - crowd_in_roi: Send alert if triggered=True and matched_count >= 3
    - restricted_zone_breach: Send alert if triggered=True
    """
    print(f"\n[SERVICE] ============== FUNCTION ENTRY: process_pipeline_alerts ==============")
    print(f"[SERVICE] Processing alerts for camera_id={request.camera_id}")
    print(f"[SERVICE] Usecase results count: {len(request.usecase_results)}")
    
    alerts_sent = []
    
    for result in request.usecase_results:
        usecase_id = result.get('usecase_id', '')
        triggered = result.get('triggered', False)
        matched_count = result.get('matched_count', 0)
        matched_objects = result.get('matched_objects', [])
        detection_id = result.get('detection_id')
        screenshot_path = result.get('screenshot_path')
        
        print(f"\n[SERVICE] Processing usecase: {usecase_id}")
        print(f"[SERVICE]   - Triggered: {triggered}")
        print(f"[SERVICE]   - Matched count: {matched_count}")
        print(f"[SERVICE]   - Detection ID: {detection_id}")
        print(f"[SERVICE]   - Screenshot path: {screenshot_path}")
        
        if not triggered:
            print(f"[SERVICE]   - Not triggered, skipping alert")
            continue
        
        # Determine alert type and whether to send
        send_alert = False
        alert_type = ""
        message = ""
        
        if usecase_id == "person_in_roi":
            send_alert = True
            alert_type = "person_detected"
            message = f"Person detected inside ROI. Count: {matched_count}"
            
        elif usecase_id == "crowd_in_roi":
            if matched_count >= 3:
                send_alert = True
                alert_type = "crowd_detected"
                message = f"Crowd detected inside ROI. Count: {matched_count}"
            else:
                print(f"[SERVICE]   - Crowd count ({matched_count}) below threshold (3), no alert")
                
        elif usecase_id == "restricted_zone_breach":
            send_alert = True
            alert_type = "restricted_zone_breach"
            message = f"Restricted zone breach detected. Count: {matched_count}"
        
        if send_alert:
            print(f"\n[ALERT] ⚠️  ALERT TRIGGERED ⚠️")
            print(f"[ALERT] Camera: {request.camera_id}")
            print(f"[ALERT] Usecase: {usecase_id}")
            print(f"[ALERT] Type: {alert_type}")
            print(f"[ALERT] Objects detected: {matched_count}")
            print(f"[ALERT] Message: {message}")
            
            alerts_sent.append(AlertDetail(
                usecase_id=usecase_id,
                alert_type=alert_type,
                alert_count=matched_count,
                message=message
            ))
            print(f"[SERVICE]   - Alert sent successfully")
            
            # Persist to database
            try:
                from shared.database.persistence import persist_alert
                persist_alert(
                    camera_id=request.camera_id,
                    usecase_name=usecase_id,
                    alert_type=alert_type,
                    status='sent',
                    detection_id=detection_id,
                    screenshot_path=screenshot_path
                )
                print(f"[SERVICE]   - Alert persisted with detection_id={detection_id}")
            except Exception as e:
                print(f"[SERVICE] DB Error: {str(e)}")
        else:
            print(f"[SERVICE]   - Alert conditions not met, no alert sent")
    
    response = PipelineAlertResponse(
        camera_id=request.camera_id,
        total_alerts_sent=len(alerts_sent),
        alerts_sent=alerts_sent
    )
    
    print(f"\n[SERVICE] Returning response: total_alerts_sent={response.total_alerts_sent}")
    print(f"[SERVICE] ============== FUNCTION EXIT: process_pipeline_alerts ==============\n")
    
    return response


def process_alert(request: AlertRequest) -> AlertResponse:
    """
    Process alert request and determine if alert should be sent (legacy function).
    
    Alert Rule:
    - Send alert if usecase_id == "person_in_roi" AND alert_required == true
    """
    print(f"\n[SERVICE] ============== FUNCTION ENTRY: process_alert ==============")
    print(f"[SERVICE] Payload received: camera_id={request.camera_id}, usecase_id={request.usecase_id}, alert_required={request.alert_required}")
    
    # Evaluate alert rule
    print(f"[SERVICE] Evaluating alert rule...")
    print(f"[SERVICE] Condition check: usecase_id == 'person_in_roi' AND alert_required == true")
    print(f"[SERVICE] usecase_id: {request.usecase_id} == 'person_in_roi': {request.usecase_id == 'person_in_roi'}")
    print(f"[SERVICE] alert_required: {request.alert_required}")
    
    alert_sent = False
    message = ""
    
    if request.usecase_id == "person_in_roi" and request.alert_required:
        # Alert rule matched - simulate sending alert
        print(f"\n[ALERT] ⚠️  ALERT TRIGGERED ⚠️")
        print(f"[ALERT] Camera: {request.camera_id}")
        print(f"[ALERT] Type: {request.alert_type}")
        print(f"[ALERT] Objects detected: {request.alert_count}")
        
        alert_sent = True
        message = "Person detected inside ROI. Alert sent."
        print(f"[SERVICE] Alert sent successfully")
    else:
        message = "Alert conditions not met. No alert sent."
        print(f"[SERVICE] Alert rule not matched - no alert sent")
    
    response = AlertResponse(
        camera_id=request.camera_id,
        alert_sent=alert_sent,
        alert_type=request.alert_type,
        alert_count=request.alert_count,
        message=message
    )
    
    print(f"[SERVICE] Returning response: alert_sent={response.alert_sent}, message={response.message}")
    print(f"[SERVICE] ============== FUNCTION EXIT: process_alert ==============\n")
    
    return response
