// her — macOS Calendar helper.
// Reads events from EventKit (much faster than AppleScript whose-clauses)
// and prints them in the same pipe-separated format the AppleScript path
// produces, so the Python parser handles both transparently.
//
// Usage: her_calendar_helper <days_ahead>
// Output: title|||calendar|||allday|||start_iso|||end_iso (one per line)

import EventKit
import Foundation

guard CommandLine.arguments.count >= 2,
      let days = Int(CommandLine.arguments[1])
else {
    FileHandle.standardError.write("usage: her_calendar_helper <days>\n".data(using: .utf8)!)
    exit(64)
}

let store = EKEventStore()
let semaphore = DispatchSemaphore(value: 0)
var grantedAccess = false
var requestError: Error?

// requestAccess(to:) signature differs across macOS versions — the
// completion-handler form is supported on 10.9+.
store.requestAccess(to: .event) { granted, error in
    grantedAccess = granted
    requestError = error
    semaphore.signal()
}
semaphore.wait()

if let err = requestError {
    FileHandle.standardError.write(
        "Calendar access error: \(err.localizedDescription)\n".data(using: .utf8)!
    )
    exit(78)
}
if !grantedAccess {
    FileHandle.standardError.write(
        "Calendar access not granted (TCC permission missing)\n".data(using: .utf8)!
    )
    exit(77)
}

let now = Date()
let cal = Calendar.current
let comps = cal.dateComponents([.year, .month, .day], from: now)
guard let startOfDay = cal.date(from: comps) else {
    FileHandle.standardError.write("date computation failed\n".data(using: .utf8)!)
    exit(70)
}
let endDate = startOfDay.addingTimeInterval(Double(days) * 86400)

let predicate = store.predicateForEvents(
    withStart: startOfDay, end: endDate, calendars: nil
)
let events = store.events(matching: predicate)

// Output in local-time ISO-ish format (matching the AppleScript path's
// "%Y-%m-%dT%H:%M:%S" without timezone).
let formatter = DateFormatter()
formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
formatter.timeZone = TimeZone.current
formatter.locale = Locale(identifier: "en_US_POSIX")

func sanitize(_ s: String) -> String {
    return s
        .replacingOccurrences(of: "|||", with: "/")
        .replacingOccurrences(of: "\n", with: " ")
        .replacingOccurrences(of: "\r", with: " ")
}

for ev in events {
    let title = sanitize(ev.title ?? "")
    let calName = sanitize(ev.calendar?.title ?? "")
    let allDay = ev.isAllDay ? "true" : "false"
    let startStr = formatter.string(from: ev.startDate)
    let endStr = ev.endDate != nil ? formatter.string(from: ev.endDate!) : ""
    print("\(title)|||\(calName)|||\(allDay)|||\(startStr)|||\(endStr)")
}
