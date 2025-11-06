from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict
from datetime import datetime
from app.db.models import BusSearchRequest, FareRequest, TravelHistory
from app.db.mongodb import mongodb
from app.utils.auth_utils import require_passenger, get_current_user
from app.external.google_maps_client import google_maps_client
from app.fares.fare_calculator import fare_calculator
from app.ai.gemini_recommender import gemini_recommender

router = APIRouter(prefix="/passenger", tags=["Passenger"])


@router.post("/search")
async def search_buses(request: BusSearchRequest, current_user: Dict = Depends(get_current_user)):
    """
    Search for buses between source and destination
    Returns available routes with ETA, fare, bus numbers, and route details
    """
    try:
        origin = (request.source_lat, request.source_lng)
        destination = (request.dest_lat, request.dest_lng)

        # Get distance and duration from Google Maps
        distance_data = google_maps_client.get_distance_matrix([origin], [destination])

        if not distance_data or "rows" not in distance_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Could not calculate distance"
            )

        element = distance_data["rows"][0]["elements"][0]
        if element["status"] != "OK":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No route found"
            )

        distance_km = element["distance"]["value"] / 1000
        duration_min = element["duration"]["value"] / 60

        # Get all active routes from database
        routes_collection = mongodb.get_collection("routes")
        buses_collection = mongodb.get_collection("buses")
        schedules_collection = mongodb.get_collection("schedules")

        all_routes = list(routes_collection.find({"status": "active"}))

        # Calculate fare
        current_hour = datetime.now().hour
        is_peak = fare_calculator.is_peak_hour(current_hour)
        fare_info = fare_calculator.calculate_fare(distance_km, is_peak)

        # Format response with bus numbers and route details
        results = []
        for route in all_routes:
            # Get buses assigned to this route
            assigned_buses = route.get("assigned_buses", [])
            bus_numbers = []

            for bus_id in assigned_buses:
                bus = buses_collection.find_one({"bus_id": bus_id})
                if bus:
                    bus_numbers.append(bus.get("bus_number", f"Bus {bus_id}"))

            # If no buses found in assigned_buses, try finding by route_id
            if not bus_numbers:
                buses_on_route = list(buses_collection.find({"assigned_route_id": route["route_id"]}))
                bus_numbers = [b.get("bus_number", f"Bus {b.get('bus_id', 'Unknown')}") for b in buses_on_route]

            # Get schedule for this route
            schedule = schedules_collection.find_one({"route_id": route["route_id"], "active": True})

            # Calculate ETA
            route_duration = route.get("estimated_duration_min", duration_min)
            eta_min = route_duration
            frequency_min = 10  # Default
            departure_times = []

            if schedule:
                frequency_min = schedule.get("frequency_min", 10)
                # Estimated wait time is half the frequency on average
                eta_min += frequency_min / 2
                departure_times = schedule.get("departure_times", [])

            # Get route details
            source_name = route.get("source_name", "Unknown")
            dest_name = route.get("dest_name", "Unknown")
            waypoints = route.get("waypoints", [])

            results.append({
                "route_id": route["route_id"],
                "route_name": route.get("name", f"{source_name} → {dest_name}"),
                "source_name": source_name,
                "dest_name": dest_name,
                "bus_numbers": bus_numbers,  # List of bus numbers on this route
                "distance_km": route.get("total_distance_km", distance_km),
                "eta_min": round(eta_min, 1),
                "fare": fare_info["final_fare"],
                "frequency_min": frequency_min,
                "is_peak_hour": is_peak,
                "waypoints": waypoints,  # Route path for map display
                "departure_times": departure_times[:5] if departure_times else [],  # Next 5 departures
                "total_buses": len(bus_numbers)
            })

        return {
            "source": {"lat": request.source_lat, "lng": request.source_lng},
            "destination": {"lat": request.dest_lat, "lng": request.dest_lng},
            "distance_km": round(distance_km, 2),
            "available_routes": results,
            "fare_details": fare_info,
            "total_routes_found": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching buses: {str(e)}"
        )


@router.get("/recommendations")
async def get_recommendations(current_user: Dict = Depends(require_passenger)):
    """
    Get personalized bus recommendations based on travel history
    """
    try:
        passenger_id = current_user.get("sub")
        
        recommendations = gemini_recommender.get_personalized_recommendations(passenger_id)
        patterns = gemini_recommender.analyze_travel_patterns(passenger_id)
        
        return {
            "recommendations": recommendations,
            "travel_patterns": patterns,
            "current_time": datetime.now().strftime("%H:%M"),
            "day": datetime.now().strftime("%A")
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )


@router.get("/history")
async def get_travel_history(current_user: Dict = Depends(require_passenger)):
    """Get passenger travel history"""
    try:
        passenger_id = current_user.get("sub")
        
        history_collection = mongodb.get_collection("travel_history")
        history = list(history_collection.find(
            {"passenger_id": passenger_id}
        ).sort("timestamp", -1).limit(50))
        
        # Convert ObjectId to string
        for record in history:
            record["_id"] = str(record["_id"])
            if isinstance(record.get("timestamp"), datetime):
                record["timestamp"] = record["timestamp"].isoformat()
        
        return {
            "total": len(history),
            "history": history
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching history: {str(e)}"
        )


@router.post("/history/log")
async def log_travel(travel_data: Dict, current_user: Dict = Depends(require_passenger)):
    """Log a travel/trip"""
    try:
        passenger_id = current_user.get("sub")
        
        now = datetime.now()
        
        history_record = {
            "passenger_id": passenger_id,
            "route_id": travel_data.get("route_id"),
            "source_stop_id": travel_data.get("source_stop_id", ""),
            "dest_stop_id": travel_data.get("dest_stop_id", ""),
            "travel_time": now.strftime("%H:%M"),
            "day_of_week": now.strftime("%A"),
            "timestamp": now
        }
        
        history_collection = mongodb.get_collection("travel_history")
        result = history_collection.insert_one(history_record)
        
        return {
            "message": "Travel logged successfully",
            "history_id": str(result.inserted_id)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error logging travel: {str(e)}"
        )


@router.post("/fare/calculate")
async def calculate_fare(request: FareRequest):
    """Calculate fare for a distance"""
    try:
        fare_info = fare_calculator.calculate_fare(
            request.distance_km,
            request.is_peak_hour
        )
        
        return fare_info
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating fare: {str(e)}"
        )


@router.get("/fare/route/{route_id}")
async def get_route_fare(route_id: str, is_peak_hour: bool = False):
    """Get fare for a specific route"""
    try:
        fare_info = fare_calculator.estimate_fare_for_route(route_id, is_peak_hour)
        
        if not fare_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        
        return fare_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting route fare: {str(e)}"
        )

