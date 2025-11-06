from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta, time
from app.db.models import RoutePlanRequest, ScheduleCreateRequest
from app.db.mongodb import mongodb
from app.utils.auth_utils import require_admin
from app.routing.graph_builder import graph_builder
from app.ai.gemini_route_optimizer import gemini_route_optimizer
from app.ai.gemini_scheduler import gemini_scheduler
from pydantic import BaseModel
import math

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])


def generate_time_slots(start_hour: int, end_hour: int, frequency_min: int) -> List[str]:
    """
    Generate time slots for bus departures
    Args:
        start_hour: Starting hour (e.g., 6 for 6:00 AM)
        end_hour: Ending hour (e.g., 22 for 10:00 PM)
        frequency_min: Frequency in minutes between buses
    Returns:
        List of time strings in HH:MM format
    """
    time_slots = []
    current_time = datetime.strptime(f"{start_hour:02d}:00", "%H:%M")
    end_time = datetime.strptime(f"{end_hour:02d}:00", "%H:%M")

    while current_time <= end_time:
        time_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=frequency_min)

    return time_slots


class BatchRoutePlanRequest(BaseModel):
    routes: List[RoutePlanRequest]


class RouteSelectionRequest(BaseModel):
    bus_id: str
    bus_number: str
    route_index: int  # Which of the 3 routes was selected (0, 1, or 2)
    source_name: str
    dest_name: str
    source_lat: float
    source_lng: float
    dest_lat: float
    dest_lng: float
    waypoints: List[List[float]]  # List of [lat, lng] pairs
    distance_km: float
    duration_min: float
    gemini_score: float
    traffic_score: float
    reasoning: str
    peak_hour: str
    expected_passengers_daily: int = 0  # For frequency calculation


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate distance between two points using Haversine formula (in km)"""
    lat1, lon1 = point1
    lat2, lon2 = point2

    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def find_route_overlaps(routes_data: List[Dict]) -> List[Dict]:
    """
    Detect where and when bus routes overlap or intersect
    Returns list of overlap points with timing information
    """
    overlaps = []

    for i in range(len(routes_data)):
        for j in range(i + 1, len(routes_data)):
            route1 = routes_data[i]
            route2 = routes_data[j]

            # Check if routes pass through similar areas (within 500m)
            # This is a simplified version - in production, use actual route polylines

            # Check source proximity
            dist_source = calculate_distance(
                (route1['source']['lat'], route1['source']['lng']),
                (route2['source']['lat'], route2['source']['lng'])
            )

            if dist_source < 0.5:  # Within 500m
                overlaps.append({
                    'bus1': route1.get('bus_number', f'Bus {i+1}'),
                    'bus2': route2.get('bus_number', f'Bus {j+1}'),
                    'location': 'Near source area',
                    'coordinates': route1['source'],
                    'distance_apart_km': round(dist_source, 2),
                    'type': 'source_overlap'
                })

            # Check destination proximity
            dist_dest = calculate_distance(
                (route1['destination']['lat'], route1['destination']['lng']),
                (route2['destination']['lat'], route2['destination']['lng'])
            )

            if dist_dest < 0.5:  # Within 500m
                overlaps.append({
                    'bus1': route1.get('bus_number', f'Bus {i+1}'),
                    'bus2': route2.get('bus_number', f'Bus {j+1}'),
                    'location': 'Near destination area',
                    'coordinates': route1['destination'],
                    'distance_apart_km': round(dist_dest, 2),
                    'type': 'destination_overlap'
                })

    return overlaps


@router.post("/route/plan")
async def plan_route(request: RoutePlanRequest):
    """
    AI-assisted route planning
    Returns top 5 optimized routes using Dijkstra + Gemini AI
    """
    try:
        origin = (request.source_lat, request.source_lng)
        destination = (request.dest_lat, request.dest_lng)

        # Get alternative routes from Google Maps
        routes = graph_builder.get_alternative_routes(origin, destination, k=5)

        if not routes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No routes found between these locations"
            )

        # Use Gemini AI to analyze and rank routes
        ranked_routes = gemini_route_optimizer.optimize_and_rank_routes(
            routes, origin, destination
        )

        return {
            "source": {"lat": request.source_lat, "lng": request.source_lng},
            "destination": {"lat": request.dest_lat, "lng": request.dest_lng},
            "total_routes": len(ranked_routes),
            "routes": ranked_routes
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error planning route: {str(e)}"
        )


@router.post("/route/plan-batch")
async def plan_batch_routes(request: BatchRoutePlanRequest):
    """
    Plan multiple bus routes simultaneously with overlap detection
    Demonstrates the power of the system by planning 10-12 routes at once
    """
    try:
        all_routes_data = []

        # Plan each route
        for idx, route_req in enumerate(request.routes):
            origin = (route_req.source_lat, route_req.source_lng)
            destination = (route_req.dest_lat, route_req.dest_lng)

            # Get alternative routes from Google Maps
            routes = graph_builder.get_alternative_routes(origin, destination, k=3)

            if routes:
                # Use Gemini AI to analyze and rank routes
                ranked_routes = gemini_route_optimizer.optimize_and_rank_routes(
                    routes, origin, destination
                )

                route_data = {
                    "bus_number": f"Bus {idx + 1}",
                    "source": {"lat": route_req.source_lat, "lng": route_req.source_lng},
                    "destination": {"lat": route_req.dest_lat, "lng": route_req.dest_lng},
                    "peak_hour": route_req.peak_hour,
                    "total_routes": len(ranked_routes),
                    "routes": ranked_routes,
                    "best_route": ranked_routes[0] if ranked_routes else None
                }
                all_routes_data.append(route_data)

        # Detect overlaps between routes
        overlaps = find_route_overlaps(all_routes_data)

        return {
            "total_buses": len(all_routes_data),
            "routes": all_routes_data,
            "overlaps": overlaps,
            "overlap_count": len(overlaps),
            "message": f"Successfully planned {len(all_routes_data)} bus routes with {len(overlaps)} overlap points detected"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error planning batch routes: {str(e)}"
        )


@router.post("/route/select-and-save")
async def select_and_save_route(request: RouteSelectionRequest, current_user: Dict = Depends(require_admin)):
    """
    Save the selected route for a specific bus with frequency calculation
    """
    try:
        routes_collection = mongodb.get_collection("routes")
        buses_collection = mongodb.get_collection("buses")

        # Generate route ID
        count = routes_collection.count_documents({})
        route_id = f"R{count + 1}"

        # Calculate bus frequency if passenger count provided
        frequency_data = None
        bus_capacity = 70  # Updated: Average bus capacity in India
        operating_hours = 16 * 60  # 16 hours in minutes
        layover_time = 10  # minutes for turnaround

        if request.expected_passengers_daily > 0:
            # SIMPLE AND CORRECT CALCULATION LOGIC
            daily_passengers = request.expected_passengers_daily
            one_way_duration = request.duration_min
            round_trip_time = (one_way_duration * 2) + layover_time

            # Step 1: How many trips needed to move all passengers?
            trips_needed = math.ceil(daily_passengers / bus_capacity)

            # Step 2: How many trips can 1 bus make in a day?
            trips_per_bus = operating_hours // round_trip_time

            # Step 3: How many buses do we need?
            buses_needed = math.ceil(trips_needed / trips_per_bus)

            # Step 4: What's the frequency? (how often does a bus depart)
            actual_frequency = round_trip_time // buses_needed

            # Step 5: Calculate actual capacity
            total_trips_per_day = trips_per_bus * buses_needed
            total_daily_capacity = total_trips_per_day * bus_capacity
            utilization_rate = (daily_passengers / total_daily_capacity) * 100

            # Use AI to get additional insights
            route_info = {
                "total_distance_km": request.distance_km,
                "estimated_duration_min": request.duration_min,
                "traffic_score": request.traffic_score
            }
            ai_prediction = gemini_scheduler.predict_schedule(route_info, request.peak_hour)

            # Combine our calculation with AI insights
            frequency_data = {
                "buses_needed": buses_needed,
                "frequency_min": actual_frequency,
                "trips_per_day": total_trips_per_day,
                "trips_needed": trips_needed,
                "trips_per_bus": trips_per_bus,
                "expected_passengers_daily": daily_passengers,
                "total_daily_capacity": total_daily_capacity,
                "utilization_rate": round(utilization_rate, 1),
                "round_trip_time": round_trip_time,
                "reasoning": f"Logic: {trips_needed} trips needed ÷ {trips_per_bus} trips/bus = {buses_needed} bus(es). Frequency: every {actual_frequency} min. Capacity: {total_daily_capacity} passengers/day ({utilization_rate:.1f}% utilization)."
            }
        else:
            # Default frequency calculation without passenger data
            route_info = {
                "total_distance_km": request.distance_km,
                "estimated_duration_min": request.duration_min,
                "traffic_score": request.traffic_score
            }
            frequency_data = gemini_scheduler.predict_schedule(route_info, request.peak_hour)

        # Create route document
        route = {
            "route_id": route_id,
            "name": f"{request.source_name} → {request.dest_name}",
            "source_name": request.source_name,
            "dest_name": request.dest_name,
            "source_coords": {"lat": request.source_lat, "lng": request.source_lng},
            "dest_coords": {"lat": request.dest_lat, "lng": request.dest_lng},
            "waypoints": request.waypoints,
            "total_distance_km": request.distance_km,
            "estimated_duration_min": request.duration_min,
            "gemini_score": request.gemini_score,
            "traffic_score": request.traffic_score,
            "reasoning": request.reasoning,
            "route_index": request.route_index,
            "peak_hour": request.peak_hour,
            "assigned_buses": [request.bus_id],
            "frequency_data": frequency_data,
            "created_by": current_user.get("email", "admin"),
            "created_at": datetime.utcnow(),
            "status": "active"
        }

        result = routes_collection.insert_one(route)
        route["_id"] = str(result.inserted_id)

        # Update or create bus document
        bus_doc = buses_collection.find_one({"bus_number": request.bus_number})
        if bus_doc:
            # Update existing bus
            buses_collection.update_one(
                {"bus_number": request.bus_number},
                {"$set": {
                    "assigned_route_id": route_id,
                    "status": "active",
                    "updated_at": datetime.utcnow()
                }}
            )
        else:
            # Create new bus
            buses_collection.insert_one({
                "bus_id": request.bus_id,
                "bus_number": request.bus_number,
                "assigned_route_id": route_id,
                "capacity": 70,  # Updated capacity
                "status": "active",
                "created_at": datetime.utcnow()
            })

        # Auto-generate schedule with time slots
        schedules_collection = mongodb.get_collection("schedules")

        # Determine operating hours based on peak hour
        if request.peak_hour == "morning":
            start_hour, end_hour = 6, 22  # 6 AM to 10 PM
        elif request.peak_hour == "evening":
            start_hour, end_hour = 6, 22
        else:  # off-peak
            start_hour, end_hour = 9, 20  # 9 AM to 8 PM

        # Generate time slots
        frequency_min = frequency_data.get("frequency_min", 10) if frequency_data else 10
        time_slots = generate_time_slots(start_hour, end_hour, frequency_min)

        # Create schedule document
        schedule = {
            "route_id": route_id,
            "bus_id": request.bus_id,
            "bus_number": request.bus_number,
            "peak_hour": request.peak_hour,
            "start_time": f"{start_hour:02d}:00",
            "end_time": f"{end_hour:02d}:00",
            "frequency_min": frequency_min,
            "suggested_buses_count": frequency_data.get("buses_needed", 1) if frequency_data else 1,
            "departure_times": time_slots,  # List of all departure times
            "active": True,
            "created_at": datetime.utcnow()
        }

        schedule_result = schedules_collection.insert_one(schedule)
        schedule["_id"] = str(schedule_result.inserted_id)

        return {
            "message": "Route and schedule saved successfully",
            "route_id": route_id,
            "route": route,
            "frequency_recommendation": frequency_data,
            "schedule": {
                "total_trips": len(time_slots),
                "first_departure": time_slots[0] if time_slots else None,
                "last_departure": time_slots[-1] if time_slots else None,
                "frequency_min": frequency_min,
                "buses_needed": frequency_data.get("buses_needed", 1) if frequency_data else 1
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving route: {str(e)}"
        )


@router.post("/route/create")
async def create_route(route_data: Dict[Any, Any], current_user: Dict = Depends(require_admin)):
    """
    Create a new bus route from planned route
    """
    routes_collection = mongodb.get_collection("routes")

    # Generate route ID
    count = routes_collection.count_documents({})
    route_id = f"R{count + 1}"

    # Create route document
    route = {
        "route_id": route_id,
        "name": route_data.get("name", f"Route {route_id}"),
        "source_stop_id": route_data.get("source_stop_id", ""),
        "dest_stop_id": route_data.get("dest_stop_id", ""),
        "path": route_data.get("path", []),
        "total_distance_km": route_data.get("distance_km", 0),
        "estimated_duration_min": route_data.get("duration_min", 0),
        "gemini_score": route_data.get("gemini_score", 0),
        "traffic_score": route_data.get("traffic_score", 0),
        "created_by": current_user.get("email", "admin"),
        "created_at": datetime.utcnow()
    }

    result = routes_collection.insert_one(route)
    route["_id"] = str(result.inserted_id)

    return {
        "message": "Route created successfully",
        "route_id": route_id,
        "route": route
    }


@router.post("/schedule/create")
async def create_schedule(request: ScheduleCreateRequest):
    """
    Create schedule with AI predictions for bus count and frequency
    """
    try:
        # Get route details
        routes_collection = mongodb.get_collection("routes")
        route = routes_collection.find_one({"route_id": request.route_id})
        
        if not route:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        
        # Use Gemini AI to predict optimal scheduling
        route_data = {
            "name": route.get("name"),
            "total_distance_km": route.get("total_distance_km"),
            "estimated_duration_min": route.get("estimated_duration_min"),
            "traffic_score": route.get("traffic_score", 7.0)
        }
        
        prediction = gemini_scheduler.predict_schedule(route_data, request.peak_hour)
        
        # Create schedule document
        schedules_collection = mongodb.get_collection("schedules")
        
        # For POC, create a simple schedule
        schedule = {
            "route_id": request.route_id,
            "bus_id": "TBD",  # To be assigned
            "peak_hour": request.peak_hour,
            "start_time": "06:00",
            "frequency_min": prediction.get("frequency_min"),
            "suggested_buses_count": prediction.get("buses_needed"),
            "active": True,
            "created_at": datetime.utcnow()
        }
        
        result = schedules_collection.insert_one(schedule)
        schedule["_id"] = str(result.inserted_id)
        
        return {
            "message": "Schedule created successfully",
            "schedule": schedule,
            "ai_prediction": prediction
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating schedule: {str(e)}"
        )


@router.get("/routes")
async def get_all_routes():
    """Get all routes"""
    routes_collection = mongodb.get_collection("routes")
    routes = list(routes_collection.find())
    
    # Convert ObjectId to string
    for route in routes:
        route["_id"] = str(route["_id"])
    
    return {
        "total": len(routes),
        "routes": routes
    }


@router.get("/routes/{route_id}")
async def get_route(route_id: str):
    """Get specific route details"""
    routes_collection = mongodb.get_collection("routes")
    route = routes_collection.find_one({"route_id": route_id})
    
    if not route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Route not found"
        )
    
    route["_id"] = str(route["_id"])
    return route


@router.get("/schedules")
async def get_all_schedules():
    """Get all schedules"""
    schedules_collection = mongodb.get_collection("schedules")
    schedules = list(schedules_collection.find())
    
    # Convert ObjectId to string
    for schedule in schedules:
        schedule["_id"] = str(schedule["_id"])
    
    return {
        "total": len(schedules),
        "schedules": schedules
    }


@router.put("/route/{route_id}")
async def update_route(route_id: str, route_data: Dict[Any, Any]):
    """Update a route"""
    routes_collection = mongodb.get_collection("routes")
    
    result = routes_collection.update_one(
        {"route_id": route_id},
        {"$set": route_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Route not found"
        )
    
    return {
        "message": "Route updated successfully",
        "route_id": route_id
    }


@router.delete("/route/{route_id}")
async def delete_route(route_id: str):
    """Delete a route"""
    routes_collection = mongodb.get_collection("routes")
    
    result = routes_collection.delete_one({"route_id": route_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Route not found"
        )
    
    return {
        "message": "Route deleted successfully",
        "route_id": route_id
    }

