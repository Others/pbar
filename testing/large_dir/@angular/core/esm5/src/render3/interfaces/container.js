/**
 * @license
 * Copyright Google Inc. All Rights Reserved.
 *
 * Use of this source code is governed by an MIT-style license that can be
 * found in the LICENSE file at https://angular.io/license
 */
import { HOST, NEXT, PARENT, T_HOST } from './view';
/**
 * Special location which allows easy identification of type. If we have an array which was
 * retrieved from the `LView` and that array has `true` at `TYPE` location, we know it is
 * `LContainer`.
 */
export var TYPE = 1;
/**
 * Below are constants for LContainer indices to help us look up LContainer members
 * without having to remember the specific indices.
 * Uglify will inline these when minifying so there shouldn't be a cost.
 */
export var ACTIVE_INDEX = 2;
// PARENT and NEXT are indices 3 and 4
// As we already have these constants in LView, we don't need to re-create them.
export var MOVED_VIEWS = 5;
// T_HOST is index 6
// We already have this constants in LView, we don't need to re-create it.
export var NATIVE = 7;
export var VIEW_REFS = 8;
/**
 * Size of LContainer's header. Represents the index after which all views in the
 * container will be inserted. We need to keep a record of current views so we know
 * which views are already in the DOM (and don't need to be re-added) and so we can
 * remove views from the DOM when they are no longer required.
 */
export var CONTAINER_HEADER_OFFSET = 9;
// Note: This hack is necessary so we don't erroneously get a circular dependency
// failure based on types.
export var unusedValueExportToPlacateAjd = 1;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udGFpbmVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vLi4vcGFja2FnZXMvY29yZS9zcmMvcmVuZGVyMy9pbnRlcmZhY2VzL2NvbnRhaW5lci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7O0dBTUc7QUFPSCxPQUFPLEVBQUMsSUFBSSxFQUFTLElBQUksRUFBRSxNQUFNLEVBQUUsTUFBTSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBR3pEOzs7O0dBSUc7QUFDSCxNQUFNLENBQUMsSUFBTSxJQUFJLEdBQUcsQ0FBQyxDQUFDO0FBQ3RCOzs7O0dBSUc7QUFDSCxNQUFNLENBQUMsSUFBTSxZQUFZLEdBQUcsQ0FBQyxDQUFDO0FBRTlCLHNDQUFzQztBQUN0QyxnRkFBZ0Y7QUFFaEYsTUFBTSxDQUFDLElBQU0sV0FBVyxHQUFHLENBQUMsQ0FBQztBQUU3QixvQkFBb0I7QUFDcEIsMEVBQTBFO0FBRTFFLE1BQU0sQ0FBQyxJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7QUFDeEIsTUFBTSxDQUFDLElBQU0sU0FBUyxHQUFHLENBQUMsQ0FBQztBQUUzQjs7Ozs7R0FLRztBQUNILE1BQU0sQ0FBQyxJQUFNLHVCQUF1QixHQUFHLENBQUMsQ0FBQztBQXNFekMsaUZBQWlGO0FBQ2pGLDBCQUEwQjtBQUMxQixNQUFNLENBQUMsSUFBTSw2QkFBNkIsR0FBRyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGUgbGljZW5zZSB0aGF0IGNhbiBiZVxuICogZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBhdCBodHRwczovL2FuZ3VsYXIuaW8vbGljZW5zZVxuICovXG5cbmltcG9ydCB7Vmlld1JlZn0gZnJvbSAnLi4vLi4vbGlua2VyL3ZpZXdfcmVmJztcblxuaW1wb3J0IHtUTm9kZX0gZnJvbSAnLi9ub2RlJztcbmltcG9ydCB7UkNvbW1lbnQsIFJFbGVtZW50fSBmcm9tICcuL3JlbmRlcmVyJztcblxuaW1wb3J0IHtIT1NULCBMVmlldywgTkVYVCwgUEFSRU5ULCBUX0hPU1R9IGZyb20gJy4vdmlldyc7XG5cblxuLyoqXG4gKiBTcGVjaWFsIGxvY2F0aW9uIHdoaWNoIGFsbG93cyBlYXN5IGlkZW50aWZpY2F0aW9uIG9mIHR5cGUuIElmIHdlIGhhdmUgYW4gYXJyYXkgd2hpY2ggd2FzXG4gKiByZXRyaWV2ZWQgZnJvbSB0aGUgYExWaWV3YCBhbmQgdGhhdCBhcnJheSBoYXMgYHRydWVgIGF0IGBUWVBFYCBsb2NhdGlvbiwgd2Uga25vdyBpdCBpc1xuICogYExDb250YWluZXJgLlxuICovXG5leHBvcnQgY29uc3QgVFlQRSA9IDE7XG4vKipcbiAqIEJlbG93IGFyZSBjb25zdGFudHMgZm9yIExDb250YWluZXIgaW5kaWNlcyB0byBoZWxwIHVzIGxvb2sgdXAgTENvbnRhaW5lciBtZW1iZXJzXG4gKiB3aXRob3V0IGhhdmluZyB0byByZW1lbWJlciB0aGUgc3BlY2lmaWMgaW5kaWNlcy5cbiAqIFVnbGlmeSB3aWxsIGlubGluZSB0aGVzZSB3aGVuIG1pbmlmeWluZyBzbyB0aGVyZSBzaG91bGRuJ3QgYmUgYSBjb3N0LlxuICovXG5leHBvcnQgY29uc3QgQUNUSVZFX0lOREVYID0gMjtcblxuLy8gUEFSRU5UIGFuZCBORVhUIGFyZSBpbmRpY2VzIDMgYW5kIDRcbi8vIEFzIHdlIGFscmVhZHkgaGF2ZSB0aGVzZSBjb25zdGFudHMgaW4gTFZpZXcsIHdlIGRvbid0IG5lZWQgdG8gcmUtY3JlYXRlIHRoZW0uXG5cbmV4cG9ydCBjb25zdCBNT1ZFRF9WSUVXUyA9IDU7XG5cbi8vIFRfSE9TVCBpcyBpbmRleCA2XG4vLyBXZSBhbHJlYWR5IGhhdmUgdGhpcyBjb25zdGFudHMgaW4gTFZpZXcsIHdlIGRvbid0IG5lZWQgdG8gcmUtY3JlYXRlIGl0LlxuXG5leHBvcnQgY29uc3QgTkFUSVZFID0gNztcbmV4cG9ydCBjb25zdCBWSUVXX1JFRlMgPSA4O1xuXG4vKipcbiAqIFNpemUgb2YgTENvbnRhaW5lcidzIGhlYWRlci4gUmVwcmVzZW50cyB0aGUgaW5kZXggYWZ0ZXIgd2hpY2ggYWxsIHZpZXdzIGluIHRoZVxuICogY29udGFpbmVyIHdpbGwgYmUgaW5zZXJ0ZWQuIFdlIG5lZWQgdG8ga2VlcCBhIHJlY29yZCBvZiBjdXJyZW50IHZpZXdzIHNvIHdlIGtub3dcbiAqIHdoaWNoIHZpZXdzIGFyZSBhbHJlYWR5IGluIHRoZSBET00gKGFuZCBkb24ndCBuZWVkIHRvIGJlIHJlLWFkZGVkKSBhbmQgc28gd2UgY2FuXG4gKiByZW1vdmUgdmlld3MgZnJvbSB0aGUgRE9NIHdoZW4gdGhleSBhcmUgbm8gbG9uZ2VyIHJlcXVpcmVkLlxuICovXG5leHBvcnQgY29uc3QgQ09OVEFJTkVSX0hFQURFUl9PRkZTRVQgPSA5O1xuXG4vKipcbiAqIFRoZSBzdGF0ZSBhc3NvY2lhdGVkIHdpdGggYSBjb250YWluZXIuXG4gKlxuICogVGhpcyBpcyBhbiBhcnJheSBzbyB0aGF0IGl0cyBzdHJ1Y3R1cmUgaXMgY2xvc2VyIHRvIExWaWV3LiBUaGlzIGhlbHBzXG4gKiB3aGVuIHRyYXZlcnNpbmcgdGhlIHZpZXcgdHJlZSAod2hpY2ggaXMgYSBtaXggb2YgY29udGFpbmVycyBhbmQgY29tcG9uZW50XG4gKiB2aWV3cyksIHNvIHdlIGNhbiBqdW1wIHRvIHZpZXdPckNvbnRhaW5lcltORVhUXSBpbiB0aGUgc2FtZSB3YXkgcmVnYXJkbGVzc1xuICogb2YgdHlwZS5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBMQ29udGFpbmVyIGV4dGVuZHMgQXJyYXk8YW55PiB7XG4gIC8qKlxuICAgKiBUaGUgaG9zdCBlbGVtZW50IG9mIHRoaXMgTENvbnRhaW5lci5cbiAgICpcbiAgICogVGhlIGhvc3QgY291bGQgYmUgYW4gTFZpZXcgaWYgdGhpcyBjb250YWluZXIgaXMgb24gYSBjb21wb25lbnQgbm9kZS5cbiAgICogSW4gdGhhdCBjYXNlLCB0aGUgY29tcG9uZW50IExWaWV3IGlzIGl0cyBIT1NULlxuICAgKi9cbiAgcmVhZG9ubHlbSE9TVF06IFJFbGVtZW50fFJDb21tZW50fExWaWV3O1xuXG4gIC8qKlxuICAgKiBUaGlzIGlzIGEgdHlwZSBmaWVsZCB3aGljaCBhbGxvd3MgdXMgdG8gZGlmZmVyZW50aWF0ZSBgTENvbnRhaW5lcmAgZnJvbSBgU3R5bGluZ0NvbnRleHRgIGluIGFuXG4gICAqIGVmZmljaWVudCB3YXkuIFRoZSB2YWx1ZSBpcyBhbHdheXMgc2V0IHRvIGB0cnVlYFxuICAgKi9cbiAgW1RZUEVdOiB0cnVlO1xuXG4gIC8qKlxuICAgKiBUaGUgbmV4dCBhY3RpdmUgaW5kZXggaW4gdGhlIHZpZXdzIGFycmF5IHRvIHJlYWQgb3Igd3JpdGUgdG8uIFRoaXMgaGVscHMgdXNcbiAgICoga2VlcCB0cmFjayBvZiB3aGVyZSB3ZSBhcmUgaW4gdGhlIHZpZXdzIGFycmF5LlxuICAgKiBJbiB0aGUgY2FzZSB0aGUgTENvbnRhaW5lciBpcyBjcmVhdGVkIGZvciBhIFZpZXdDb250YWluZXJSZWYsXG4gICAqIGl0IGlzIHNldCB0byBudWxsIHRvIGlkZW50aWZ5IHRoaXMgc2NlbmFyaW8sIGFzIGluZGljZXMgYXJlIFwiYWJzb2x1dGVcIiBpbiB0aGF0IGNhc2UsXG4gICAqIGkuZS4gcHJvdmlkZWQgZGlyZWN0bHkgYnkgdGhlIHVzZXIgb2YgdGhlIFZpZXdDb250YWluZXJSZWYgQVBJLlxuICAgKi9cbiAgW0FDVElWRV9JTkRFWF06IG51bWJlcjtcblxuICAvKipcbiAgICogQWNjZXNzIHRvIHRoZSBwYXJlbnQgdmlldyBpcyBuZWNlc3Nhcnkgc28gd2UgY2FuIHByb3BhZ2F0ZSBiYWNrXG4gICAqIHVwIGZyb20gaW5zaWRlIGEgY29udGFpbmVyIHRvIHBhcmVudFtORVhUXS5cbiAgICovXG4gIFtQQVJFTlRdOiBMVmlldztcblxuICAvKipcbiAgICogVGhpcyBhbGxvd3MgdXMgdG8ganVtcCBmcm9tIGEgY29udGFpbmVyIHRvIGEgc2libGluZyBjb250YWluZXIgb3IgY29tcG9uZW50XG4gICAqIHZpZXcgd2l0aCB0aGUgc2FtZSBwYXJlbnQsIHNvIHdlIGNhbiByZW1vdmUgbGlzdGVuZXJzIGVmZmljaWVudGx5LlxuICAgKi9cbiAgW05FWFRdOiBMVmlld3xMQ29udGFpbmVyfG51bGw7XG5cbiAgLyoqXG4gICAqIEEgY29sbGVjdGlvbiBvZiB2aWV3cyBjcmVhdGVkIGJhc2VkIG9uIHRoZSB1bmRlcmx5aW5nIGA8bmctdGVtcGxhdGU+YCBlbGVtZW50IGJ1dCBpbnNlcnRlZCBpbnRvXG4gICAqIGEgZGlmZmVyZW50IGBMQ29udGFpbmVyYC4gV2UgbmVlZCB0byB0cmFjayB2aWV3cyBjcmVhdGVkIGZyb20gYSBnaXZlbiBkZWNsYXJhdGlvbiBwb2ludCBzaW5jZVxuICAgKiBxdWVyaWVzIGNvbGxlY3QgbWF0Y2hlcyBmcm9tIHRoZSBlbWJlZGRlZCB2aWV3IGRlY2xhcmF0aW9uIHBvaW50IGFuZCBfbm90XyB0aGUgaW5zZXJ0aW9uIHBvaW50LlxuICAgKi9cbiAgW01PVkVEX1ZJRVdTXTogTFZpZXdbXXxudWxsO1xuXG4gIC8qKlxuICAgKiBQb2ludGVyIHRvIHRoZSBgVE5vZGVgIHdoaWNoIHJlcHJlc2VudHMgdGhlIGhvc3Qgb2YgdGhlIGNvbnRhaW5lci5cbiAgICovXG4gIFtUX0hPU1RdOiBUTm9kZTtcblxuICAvKiogVGhlIGNvbW1lbnQgZWxlbWVudCB0aGF0IHNlcnZlcyBhcyBhbiBhbmNob3IgZm9yIHRoaXMgTENvbnRhaW5lci4gKi9cbiAgcmVhZG9ubHlbTkFUSVZFXTpcbiAgICAgIFJDb21tZW50OyAgLy8gVE9ETyhtaXNrbyk6IHJlbW92ZSBhcyB0aGlzIHZhbHVlIGNhbiBiZSBnb3R0ZW4gYnkgdW53cmFwcGluZyBgW0hPU1RdYFxuXG4gIC8qKlxuICAgKiBBcnJheSBvZiBgVmlld1JlZmBzIHVzZWQgYnkgYW55IGBWaWV3Q29udGFpbmVyUmVmYHMgdGhhdCBwb2ludCB0byB0aGlzIGNvbnRhaW5lci5cbiAgICpcbiAgICogVGhpcyBpcyBsYXppbHkgaW5pdGlhbGl6ZWQgYnkgYFZpZXdDb250YWluZXJSZWZgIHdoZW4gdGhlIGZpcnN0IHZpZXcgaXMgaW5zZXJ0ZWQuXG4gICAqL1xuICBbVklFV19SRUZTXTogVmlld1JlZltdfG51bGw7XG59XG5cbi8vIE5vdGU6IFRoaXMgaGFjayBpcyBuZWNlc3Nhcnkgc28gd2UgZG9uJ3QgZXJyb25lb3VzbHkgZ2V0IGEgY2lyY3VsYXIgZGVwZW5kZW5jeVxuLy8gZmFpbHVyZSBiYXNlZCBvbiB0eXBlcy5cbmV4cG9ydCBjb25zdCB1bnVzZWRWYWx1ZUV4cG9ydFRvUGxhY2F0ZUFqZCA9IDE7XG4iXX0=